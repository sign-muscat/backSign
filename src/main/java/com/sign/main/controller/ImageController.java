package com.sign.main.controller;

// import org.apache.http.client.methods.CloseableHttpResponse;
// import org.apache.http.client.methods.HttpPost;
// import org.apache.http.entity.ContentType;
// import org.apache.http.entity.mime.MultipartEntityBuilder;
// import org.apache.http.impl.client.CloseableHttpClient;
// import org.apache.http.impl.client.HttpClients;
// import org.apache.http.util.EntityUtils;
// import org.springframework.beans.factory.annotation.Value;
// import org.springframework.http.HttpStatus;
// import org.springframework.http.ResponseEntity;
// import org.springframework.web.bind.annotation.*;
// import org.springframework.web.multipart.MultipartFile;

// import java.io.IOException;

// @RestController
// @RequestMapping("/api")
// public class ImageController {

//     @Value("${fastapi.url}")
//     private String fastApiUrl;

//     @PostMapping("/upload")
//     public ResponseEntity<String> uploadImage(@RequestParam("file") MultipartFile file, @RequestParam("wordNo") int wordNo) {
//         try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
//             HttpPost uploadFile = new HttpPost(fastApiUrl + "/upload/");
//             MultipartEntityBuilder builder = MultipartEntityBuilder.create();
//             builder.addBinaryBody("file", file.getInputStream(), ContentType.APPLICATION_OCTET_STREAM, file.getOriginalFilename());
//             builder.addTextBody("wordNo", String.valueOf(wordNo), ContentType.TEXT_PLAIN);
//             uploadFile.setEntity(builder.build());

//             try (CloseableHttpResponse response = httpClient.execute(uploadFile)) {
//                 String responseString = EntityUtils.toString(response.getEntity(), "UTF-8");
//                 return new ResponseEntity<>(responseString, HttpStatus.valueOf(response.getStatusLine().getStatusCode()));
//             }
//         } catch (IOException e) {
//             return new ResponseEntity<>(e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
//         }
//     }
// }



import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api")
public class ImageController {

    @Value("${fastapi.url}")
    private String fastApiUrl;

    @PostMapping("/upload")
    public ResponseEntity<String> uploadImage(@RequestParam("file") MultipartFile file, @RequestParam("wordNo") int wordNo) {
        try {
            String response = sendImageToFastApi(file, wordNo);
            return ResponseEntity.status(HttpStatus.OK).body(response);
        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Failed to upload image: " + e.getMessage());
        }
    }

    private String sendImageToFastApi(MultipartFile file, int wordNo) throws IOException {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpPost uploadFile = new HttpPost(fastApiUrl + "/upload/");
            MultipartEntityBuilder builder = MultipartEntityBuilder.create();
            builder.addBinaryBody("file", file.getInputStream(), ContentType.APPLICATION_OCTET_STREAM, file.getOriginalFilename());
            builder.addTextBody("wordNo", String.valueOf(wordNo), ContentType.TEXT_PLAIN);
            uploadFile.setEntity(builder.build());

            try (CloseableHttpResponse response = httpClient.execute(uploadFile)) {
                return EntityUtils.toString(response.getEntity(), "UTF-8");
            }
        }
    }
}

