import re
import sys
import urllib
import requests


def getPage(keyword,page,n):
    page=page*n
    keyword=urllib.parse.quote(keyword, safe='/')
    url_begin= "http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word="
    url = url_begin+ keyword + "&pn=" +str(page) + "&gsm="+str(hex(page))+"&ct=&ic=0&lm=-1&width=0&height=0"
    return url

def get_onepage_urls(onepageurl):
    try:
        html = requests.get(onepageurl).text
    except Exception as e:
        print(e)
        pic_urls = []
        return pic_urls
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    return pic_urls

def get_url_one_page(url):
    html = requests.get(url)
    html.encoding = 'utf-8'
    html = html.text
    url_pic_this_page = re.findall(r'"objURL":"(.*?)",', html)
    url_next_page_prefix = re.findall(r'<a href="(.*?)" class="n">下一页', html)
    if len(url_next_page_prefix) != 0:
        url_next_page = 'http://image.baidu.com' + url_next_page_prefix[0]
    else:
        print("已到达最后一页！")
        url_next_page = None
    return url_pic_this_page, url_next_page

def down_pic(pic_urls, picPath):
    """给出图片链接列表, 下载所有图片"""
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            # string =str(i + 1) + '.jpg'
            string = picPath + "/" + str(i + 1) + '.jpg'
            with open(string, 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue


if __name__ == '__main__':
    urltb = "https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=ps%E5%89%8D%E5%90%8E"
    keyword = 'ps前后'  # 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
    page_begin = 0
    page_number = 20
    image_number = 3
    savedir = "D:/data/imgs/makeup/all/baidu"
    all_pic_urls = []
    while 1:
        if page_begin > image_number:
            break
        print("第%d次请求数据", [page_begin])
        url = getPage(keyword, page_begin, page_number)
        pics, nextpage = get_url_one_page(urltb)

        onepage_urls = get_onepage_urls(url)
        page_begin += 1

        all_pic_urls.extend(onepage_urls)

    down_pic(list(set(all_pic_urls)), savedir)

# def get_onepage_urls(onepageurl):
#     """获取单个翻页的所有图片的urls+当前翻页的下一翻页的url"""
#     if not onepageurl:
#         print('已到最后一页, 结束')
#         return [], ''
#     try:
#         html = requests.get(onepageurl).text
#     except Exception as e:
#         print(e)
#         pic_urls = []
#         fanye_url = ''
#         return pic_urls, fanye_url
#     pic_urls = re.findall('"ObjUrl":"(.*?)",', html, re.S)
#     fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
#     fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
#     return pic_urls, fanye_url
#
#
# def down_pic(pic_urls, picPath):
#     """给出图片链接列表, 下载所有图片"""
#     for i, pic_url in enumerate(pic_urls):
#         try:
#             pic = requests.get(pic_url, timeout=15)
#             string = picPath + "/" + str(i + 1) + '.jpg'
#             with open(string, 'wb') as f:
#                 f.write(pic.content)
#                 print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
#         except Exception as e:
#             print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
#             print(e)
#             continue
#
#
# if __name__ == '__main__':
#     keyword = '淡妆 抖音'  # 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
#     # url_init_first = r'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1577696127647_R&pv=&ic=0&nc=1&z=0&hd=0&latest=0&copyright=0&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word='
#     url_init_first = "http://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word="
#     url_init = url_init_first + urllib.parse.quote(keyword, safe='/')
#     all_pic_urls = []
#     onepage_urls, fanye_url = get_onepage_urls(url_init)
#     all_pic_urls.extend(onepage_urls)
#
#     fanye_count = 0  # 累计翻页数
#     while 1:
#         onepage_urls, fanye_url = get_onepage_urls(fanye_url)
#         fanye_count += 1
#         print('第%s页' % fanye_count)
#         if fanye_url == '' and onepage_urls == []:
#             break
#         all_pic_urls.extend(onepage_urls)
#
#     savePath = "D:/data/imgs/makeup/all/baidu"
#     down_pic(list(set(all_pic_urls)), savePath)