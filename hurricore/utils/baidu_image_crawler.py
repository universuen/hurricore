from __future__ import annotations  

import os
import time
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm


from hurricore.utils import get_random_user_agent


class BaiduImageCrawler:
    def __init__(
        self,
        num_threads: int = os.cpu_count(),
        interval: float = 0.5,
        time_out: float = 5,
    ):
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.interval = interval
        self.time_out = time_out
    

    def crawl_image_urls(self, query: str, num_image_urls: int = 10000) -> list[str]:
        query_encoded = urllib.parse.quote(query)
        num_image_urls_per_page = 30
        num_crawled = 0
        image_urls = []
        p_bar = tqdm(total=num_image_urls, desc="Crawling image urls")
        while num_crawled < num_image_urls:
            url = (
                "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result"
                f"&queryWord={query_encoded}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright="
                f"&word={query_encoded}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force="
                f"&pn={num_crawled}&rn={num_image_urls_per_page}&gsm=1e&1594447993172="
            )
            response = requests.get(
                url=url,
                headers={
                    "User-Agent": get_random_user_agent(),
                    "Accept-language": "zh-CN,zh;q=0.9"
                }
            ).content
            time.sleep(self.interval)

            try:
                respons_json = json.loads(response, strict=False)
            except json.decoder.JSONDecodeError:
                num_crawled += num_image_urls_per_page
                p_bar.update(num_image_urls_per_page)
                continue
            if "data" not in respons_json:
                continue

            image_urls.extend(self._get_sub_image_urls(respons_json))
            num_crawled += num_image_urls_per_page
            p_bar.update(num_image_urls_per_page)
            
        return list(set(image_urls))
    

    def _get_sub_image_urls(self, respons_json: dict) -> list[str]:
        sub_image_urls = []
        for image_info in respons_json['data']:
            if 'replaceUrl' not in image_info or len(image_info['replaceUrl']) < 1:
                continue
            obj_url = image_info['replaceUrl'][0]['ObjUrl']
            thumb_url = image_info['thumbURL']
            url = (
                "https://image.baidu.com/search/down?tn=download&ipn=dwnl&word=download&ie=utf8&fr=result"
                f"&url={urllib.parse.quote(obj_url)}&thumburl={urllib.parse.quote(thumb_url)}"
            )
            sub_image_urls.append(url)
        return sub_image_urls
