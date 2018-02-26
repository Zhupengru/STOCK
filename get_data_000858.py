# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:09:46 2018

@author: 79057
"""

# -*- coding: utf-8 -*-
#import scrapy

from scrapy.spider import BaseSpider
from stockdata.settings import STOCK_CODE

class ToScrapeSpiderXPath(BaseSpider):
    name = 'getdata'
    allows_domains = ['http://www.aigoogoo.com']
    
    start_urls = (
        'http://www.aigaogao.com/tools/history.html?s='+STOCK_CODE,
    )

    def parse(self, response):
        for i in range(2662,2,-1):
            yield {
                'open': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[2]/text()').extract(),
                'close': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[5]/text()').extract(),
                'low': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[4]/text()').extract(),
                'high': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[3]/text()').extract(),
                'volume': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[6]/text()').extract(),
                'money': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[7]/text()').extract(),
                'change': response.xpath('//*[@id="ctl16_contentdiv"]/table/tr['+str(i)+']/td[9]/span/text()').extract()
            }

'''           
open,close,low,high,volume,money,change
2662
open://*[@id="ctl16_contentdiv"]/table/tr[2]/td[2]
close://*[@id="ctl16_contentdiv"]/table/tr[2]/td[5]
low://*[@id="ctl16_contentdiv"]/table/tr[2]/td[4]
high://*[@id="ctl16_contentdiv"]/table/tr[2]/td[3]
volume://*[@id="ctl16_contentdiv"]/table/tr[2]/td[6]
money://*[@id="ctl16_contentdiv"]/table/tr[2]/td[7]
change://*[@id="ctl16_contentdiv"]/table/tr[2]/td[9]/span
'''


//*[@id="ctl16_contentdiv"]/table/tbody/tr[2]/td[2]