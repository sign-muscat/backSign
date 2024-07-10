package com.sign.main.controller;

import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.GetMapping;


@RestController
public class TestController {
    
    @GetMapping("/api/test")
    public void test() {
        System.out.println("테스트 파일");
    }
    
}
