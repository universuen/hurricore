from hurricore.utils import BaiduImageCrawler


def test_bing_image_crawler():
    crawler = BaiduImageCrawler()
    urls = crawler.crawl_image_urls('打火机', num_image_urls=100)
    for url in urls[:10]:
        print(url)
    assert len(urls) > 0
    assert isinstance(urls[0], str), 'Image urls should be strings'


if __name__ == '__main__':
    test_bing_image_crawler()
