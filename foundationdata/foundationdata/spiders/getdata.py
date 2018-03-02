# -*- coding: utf-8 -*-
import scrapy
from foundationdata.settings import STOCK_CODE 



class GetdataSpider(scrapy.Spider):
    name = 'getdata'
    allowed_domains = ['http://www.aigoogoo.com']
    start_urls = ['http://www.aigaogao.com/tools/history.html?s='+STOCK_CODE,]

    def parse(self, response):
        for i in range(2663,2,-1): 
            yield {
                'date':response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[1]/a/text()').extract(),
                'open': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[2]/text()').extract(), 
                'close': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[5]/text()').extract(), 
                'low': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[4]/text()').extract(), 
                'high': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[3]/text()').extract(), 
                'volume': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[6]/text()').extract(), 
                'money': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[7]/text()').extract(), 
                'change': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[9]/span/text()').extract(),
                'label': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i-1)+']/td[5]/text()').extract()
            }