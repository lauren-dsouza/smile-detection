/*********
 PlatformIO-compatible face + smile detection
 - No ESP-WHO required
 - Uses simple skin-color blob detection + mouth ratio
 - Streams MJPEG like original code
*********/


#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h" //disable brownout problems
#include "soc/rtc_cntl_reg.h"  //disable brownout problems
#include "esp_http_server.h"
#include <stdlib.h>
#include <string.h>


#define SETUP_AP 1 // 1=AP, 0=STA


const char* ssid = "ESP32_3093689";
const char* password = "123456789";


#define PART_BOUNDARY "123456789000000000000987654321"


#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM
#include "camera_pins.h"

// <<< ADDED: LED pin for XIAO ESP32-S3
#define LED_PIN 21


// detection settings
#define DETECT_W 160   // width to downscale to for detection
#define DETECT_H 120   // height to downscale to for detection
#define SKIN_MIN_CB 77 // Cb/Cr thresholds for skin (YCbCr)
#define SKIN_MAX_CB 127
#define SKIN_MIN_CR 133
#define SKIN_MAX_CR 173


// smile threshold (tune if needed)
#define SMILE_THRESHOLD 1.25f


static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";


httpd_handle_t stream_httpd = NULL;


// helper: clamp
static inline int clampi(int v, int a, int b){ if(v<a) return a; if(v>b) return b; return v; }


// Convert RGB888 buffer to YCbCr and check skin-range
// rgb points to width*height*3 (R,G,B)
static void skin_map_from_rgb(uint8_t *rgb, int w, int h, uint8_t *out_mask){
 for(int y=0;y<h;y++){
   for(int x=0;x<w;x++){
     int idx = (y*w + x)*3;
     uint8_t r = rgb[idx+0];
     uint8_t g = rgb[idx+1];
     uint8_t b = rgb[idx+2];
     // convert to YCbCr (BT.601) approximate
     int Y  = ( ( 66*r + 129*g +  25*b + 128) >> 8) + 16;
     int Cb = ( (-38*r -  74*g + 112*b + 128) >> 8) + 128;
     int Cr = ( (112*r -  94*g -  18*b + 128) >> 8) + 128;
     // skin range test (simple)
     if (Cb >= SKIN_MIN_CB && Cb <= SKIN_MAX_CB && Cr >= SKIN_MIN_CR && Cr <= SKIN_MAX_CR) {
       out_mask[y*w + x] = 1;
     } else {
       out_mask[y*w + x] = 0;
     }
   }
 }
}


// Fast bounding-box of largest skin blob via simple connected-component-ish scan using flood fill on downscaled mask
static bool find_largest_skin_bbox(uint8_t *mask, int w, int h, int &bx1, int &by1, int &bx2, int &by2){
 // We'll find the largest blob by scanning and performing a simple BFS flood fill using a small queue.
 // To keep memory small, we mark visited by setting mask to 2.
 int max_area = 0;
 int max_x1=0, max_y1=0, max_x2=0, max_y2=0;
 int* qx = (int*)malloc(w*h*sizeof(int));
 int* qy = (int*)malloc(w*h*sizeof(int));
 if(!qx || !qy) {
   if(qx) free(qx);
   if(qy) free(qy);
   return false;
 }


 for(int y=0;y<h;y++){
   for(int x=0;x<w;x++){
     int idx = y*w + x;
     if(mask[idx] == 1){
       // flood fill
       int qhead=0, qtail=0;
       qx[qtail]=x; qy[qtail]=y; qtail++;
       mask[idx] = 2;
       int area = 0;
       int minx=x, miny=y, maxx=x, maxy=y;
       while(qhead<qtail){
         int cx = qx[qhead];
         int cy = qy[qhead];
         qhead++;
         area++;
         if(cx<minx) minx=cx;
         if(cx>maxx) maxx=cx;
         if(cy<miny) miny=cy;
         if(cy>maxy) maxy=cy;
         // 4-neighbors
         const int nx[4] = {1,-1,0,0};
         const int ny[4] = {0,0,1,-1};
         for(int k=0;k<4;k++){
           int nxp = cx + nx[k];
           int nyp = cy + ny[k];
           if(nxp>=0 && nxp<w && nyp>=0 && nyp<h){
             int nidx = nyp*w + nxp;
             if(mask[nidx] == 1){
               mask[nidx] = 2;
               qx[qtail]=nxp; qy[qtail]=nyp; qtail++;
             }
           }
         }
       }
       if(area > max_area){
         max_area = area;
         max_x1 = minx; max_y1 = miny; max_x2 = maxx; max_y2 = maxy;
       }
     }
   }
 }


 free(qx); free(qy);


 if(max_area == 0) return false;
 bx1 = max_x1; by1 = max_y1; bx2 = max_x2; by2 = max_y2;
 return true;
}


// Compute simple smile score inside lower third of face bbox
// We use bright/dark horizontal structure: mouth gap tends to be a dark horizontal band and wider when smiling.
static float compute_smile_score_from_rgb(uint8_t *rgb, int img_w, int img_h, int bx1, int by1, int bx2, int by2){
 int face_w = bx2 - bx1 + 1;
 int face_h = by2 - by1 + 1;
 if(face_w <= 4 || face_h <= 4) return 0.0f;


 int mouth_y1 = by1 + (face_h * 2 / 3);
 int mouth_y2 = by2;
 mouth_y1 = clampi(mouth_y1, by1, by2);
 mouth_y2 = clampi(mouth_y2, by1, by2);


 // Convert mouth area to grayscale and compute horizontal projection of dark pixels
 int width = img_w;
 int rows = mouth_y2 - mouth_y1 + 1;
 if(rows <= 0) return 0.0f;


 // For each row, count dark pixels across face width
 int max_dark_width = 0;
 int dark_row_count = 0;
 for(int y = mouth_y1; y <= mouth_y2; y++){
   int row_dark = 0;
   for(int x = bx1; x <= bx2; x++){
     int idx = (y * width + x) * 3;
     uint8_t r = rgb[idx+0];
     uint8_t g = rgb[idx+1];
     uint8_t b = rgb[idx+2];
     uint8_t gray = (uint8_t)((uint16_t)r*30/100 + (uint16_t)g*59/100 + (uint16_t)b*11/100);
     // threshold for dark pixel - mouth gap is darker
     if(gray < 90) row_dark++;
   }
   if(row_dark > max_dark_width) max_dark_width = row_dark;
   if(row_dark > (face_w/8)) dark_row_count++;
 }


 // vertical measure: how many rows have a significant dark width
 float vertical_extent = (float)dark_row_count / (float)rows; // 0..1
 // horizontal measure: normalized dark width
 float horizontal_extent = (float)max_dark_width / (float)face_w; // 0..1


 // smile score: ratio emphasizing horizontal widening vs vertical thickness
 // When smiling, horizontal_extent increases more than vertical_extent
 if(vertical_extent == 0.0f) return horizontal_extent * 0.5f;
 float score = (horizontal_extent / (vertical_extent + 0.001f)) * 1.0f;


 // scale to a more usable range
 score *= 2.0f;
 return score;
}


static esp_err_t stream_handler(httpd_req_t *req){
 camera_fb_t * fb = NULL;
 esp_err_t res = ESP_OK;
 size_t _jpg_buf_len = 0;
 uint8_t * _jpg_buf = NULL;
 char * part_buf[64];


 res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
 if(res != ESP_OK){
   return res;
 }


 // allocate temporary buffers for detection (downscaled)
 const int DW = DETECT_W;
 const int DH = DETECT_H;
 uint8_t *down_rgb = (uint8_t*)malloc(DW * DH * 3);
 uint8_t *skin_mask = (uint8_t*)malloc(DW * DH);
 if(!down_rgb || !skin_mask){
   if(down_rgb) free(down_rgb);
   if(skin_mask) free(skin_mask);
   Serial.println("Allocation failed for detection buffers");
   // continue with streaming but no detection
 }


 while(true){
   fb = esp_camera_fb_get();
   if (!fb) {
     Serial.println("Camera capture failed");
     res = ESP_FAIL;
   } else {
     // only attempt smile detection if we successfully allocated buffers
     if(down_rgb && skin_mask) {
       // Convert JPEG to full-size RGB into a temporary buffer of original size
       // We'll use fmt2rgb888 to write into a temp buffer sized fb->width*fb->height*3,
       // then downscale into down_rgb.
       uint8_t *full_rgb = (uint8_t*)malloc(fb->width * fb->height * 3);
       if(full_rgb){
         if(fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, full_rgb) == 0){
           // downscale nearest-neighbor to DWxDH
           for(int y=0;y<DH;y++){
             int sy = (y * fb->height) / DH;
             for(int x=0;x<DW;x++){
               int sx = (x * fb->width) / DW;
               int sidx = (sy * fb->width + sx)*3;
               int didx = (y * DW + x)*3;
               down_rgb[didx+0] = full_rgb[sidx+0];
               down_rgb[didx+1] = full_rgb[sidx+1];
               down_rgb[didx+2] = full_rgb[sidx+2];
             }
           }


           // skin map
           skin_map_from_rgb(down_rgb, DW, DH, skin_mask);


           // find largest skin blob bbox
           int bx1,by1,bx2,by2;
           // copy mask because find_largest_skin_bbox will modify it
           uint8_t *mask_copy = (uint8_t*)malloc(DW*DH);
           if(mask_copy){
             memcpy(mask_copy, skin_mask, DW*DH);
             bool found = find_largest_skin_bbox(mask_copy, DW, DH, bx1, by1, bx2, by2);
             free(mask_copy);
             if(found){
               // map bbox back to full resolution coordinates
               int full_x1 = (bx1 * fb->width) / DW;
               int full_y1 = (by1 * fb->height) / DH;
               int full_x2 = (bx2 * fb->width) / DW;
               int full_y2 = (by2 * fb->height) / DH;
               // compute smile score using full_rgb (better precision)
               float smile_score = compute_smile_score_from_rgb(full_rgb, fb->width, fb->height, full_x1, full_y1, full_x2, full_y2);


               Serial.println("FACE DETECTED!");
               Serial.print("Smile Score: ");
               Serial.println(smile_score);


               if(smile_score > SMILE_THRESHOLD){
                 Serial.println("😊 SMILE DETECTED!");
                 // <<< ADDED: turn LED OFF on smile
                 digitalWrite(LED_PIN, LOW);
               } else {
                 Serial.println("😐 Not smiling");
                 // <<< ADDED: turn LED ON when not smiling
                 digitalWrite(LED_PIN, HIGH);
               }
             }
           }
         } else {
           // fmt2rgb888 returned non-zero => conversion failed
           // we skip detection
         }
         free(full_rgb);
       } // end if full_rgb
     } // end if buffers


     // Continue original streaming logic (send JPEG)
     if(fb->width > 400){
       if(fb->format != PIXFORMAT_JPEG){
         bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
         esp_camera_fb_return(fb);
         fb = NULL;
         if(!jpeg_converted){
           Serial.println("JPEG compression failed");
           res = ESP_FAIL;
         }
       } else {
         _jpg_buf_len = fb->len;
         _jpg_buf = fb->buf;
       }
     }
   }


   if(res == ESP_OK){
     size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
     res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
   }
   if(res == ESP_OK){
     res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
   }
   if(res == ESP_OK){
     res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
   }
   if(fb){
     esp_camera_fb_return(fb);
     fb = NULL;
     _jpg_buf = NULL;
   } else if(_jpg_buf){
     free(_jpg_buf);
     _jpg_buf = NULL;
   }
   if(res != ESP_OK){
     break;
   }
 } // end while


 if(down_rgb) free(down_rgb);
 if(skin_mask) free(skin_mask);


 return res;
}


void startCameraServer(){
 httpd_config_t config = HTTPD_DEFAULT_CONFIG();
 config.server_port = 80;


 httpd_uri_t index_uri = {
   .uri       = "/",
   .method    = HTTP_GET,
   .handler   = stream_handler,
   .user_ctx  = NULL
 };
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
   httpd_register_uri_handler(stream_httpd, &index_uri);
 }
}


void setup() {
 WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); //disable brownout detector
 Serial.begin(115200);


 while(!Serial) {
   static int retries = 0;
   delay(1000); // Wait for serial monitor to open
   if (retries++ > 5) {
     break;
   }
 }
  Serial.setDebugOutput(false);

  // <<< ADDED: configure LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH); // LED ON by default (no smile)

  camera_config_t config;
 config.ledc_channel = LEDC_CHANNEL_0;
 config.ledc_timer = LEDC_TIMER_0;
 config.pin_d0 = Y2_GPIO_NUM;
 config.pin_d1 = Y3_GPIO_NUM;
 config.pin_d2 = Y4_GPIO_NUM;
 config.pin_d3 = Y5_GPIO_NUM;
 config.pin_d4 = Y6_GPIO_NUM;
 config.pin_d5 = Y7_GPIO_NUM;
 config.pin_d6 = Y8_GPIO_NUM;
 config.pin_d7 = Y9_GPIO_NUM;
 config.pin_xclk = XCLK_GPIO_NUM;
 config.pin_pclk = PCLK_GPIO_NUM;
 config.pin_vsync = VSYNC_GPIO_NUM;
 config.pin_href = HREF_GPIO_NUM;
 config.pin_sccb_sda = SIOD_GPIO_NUM;
 config.pin_sccb_scl = SIOC_GPIO_NUM;
 config.pin_pwdn = PWDN_GPIO_NUM;
 config.pin_reset = RESET_GPIO_NUM;
 config.xclk_freq_hz = 20000000;
 config.frame_size = FRAMESIZE_VGA;
 config.pixel_format = PIXFORMAT_JPEG; // for streaming
 config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
 config.fb_location = CAMERA_FB_IN_PSRAM;
 config.jpeg_quality = 12;
 config.fb_count = 1;
  // Camera init
 esp_err_t err = esp_camera_init(&config);
 if (err != ESP_OK) {
   Serial.printf("Camera init failed with error 0x%x", err);
   return;
 }


 Serial.printf("Camera init success!\n");
 Serial.printf("frame_size=%d\n", config.frame_size);
 Serial.printf("pixel_format=%d\n", config.pixel_format);


 // Wi-Fi connection
 #if SETUP_AP==1
   WiFi.softAP(ssid, password);
   Serial.print("Camera Stream Ready! Go to: http://");
   Serial.print(WiFi.softAPIP());
 #else
   WiFi.begin(ssid, password);
   while (WiFi.status() != WL_CONNECTED) {
     delay(500);
     Serial.print(".");
   }
   Serial.println("");
   Serial.println("WiFi connected");
   Serial.print("Camera Stream Ready! Go to: http://");
   Serial.print(WiFi.localIP());
 #endif
  // Start streaming web server
 startCameraServer();
}


void loop() {
 delay(1000);
 Serial.print(".");
}
