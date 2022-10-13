"""
Run by the evaluator, tries to make a GET request to a given server
"""

import argparse
import logging
import os
import random
import socket
import sys
import time
import traceback
import urllib.request

import requests

#socket.setdefaulttimeout(1)
socket.setdefaulttimeout(0.5)

import external_sites
import actions.utils

from plugins.plugin_client import ClientPlugin

BASEPATH = os.path.dirname(os.path.abspath(__file__))


class HTTPClient(ClientPlugin):
    """
    Defines the HTTP client.
    """
    name = "http"

    def __init__(self, args):
        """
        Initializes the HTTP client.
        """
        ClientPlugin.__init__(self)
        self.args = args

    @staticmethod
    def get_args(command):
        """
        Defines required args for this plugin
        """
        super_args = ClientPlugin.get_args(command)
        parser = argparse.ArgumentParser(description='HTTP Client', prog="http/client.py")

        parser.add_argument('--host-header', action='store', default="", help='specifies host header for HTTP request')
        parser.add_argument('--injected-http-contains', action='store', default="", help='checks if injected http response contains string')

        args, _ = parser.parse_known_args(command)
        args = vars(args)

        super_args.update(args)
        return super_args

    def run(self, args, logger, engine=None):
        """
        Try to make a forbidden GET request to the server.
        """
        fitness = 0
        url = args.get("server", "") # url=192.168.112.129
        assert url, "Cannot launch HTTP test with no server"
        if not url.startswith("http://"):
            url = "http://" + url # url=http://192.168.112.129
        headers = {}
        if args.get('host_header'):
            headers["Host"] = args.get('host_header')

        # If we've been given a non-standard port, append that to the URL
        port = args.get("port", 80) # port =8080
        if port != 80:
            url += ":%s" % str(port) # url=http://192.168.112.129:8080

        #if args.get("bad_word"):
        #    url += "?q=%s" % args.get("bad_word") # url =http://192.168.112.129:8080?q=hello
        if args.get("bad_word"):
            url += "/%s" % args.get("bad_word") # url =http://192.168.112.129:8080/hello.html


        injected_http = args.get("injected_http_contains")
        """
        while True:
            with open('/mnt/hgfs/share-folders/lock','r') as f:
                lock=f.read(1)
                if lock=='1':
                    break
        """
        try:
            #res = requests.get(url, allow_redirects=False, timeout=3, headers=headers) # 会被捕获然后从另外的地方发送过去 engine发送的
            req = urllib.request.Request(url, headers=headers)
            res = urllib.request.urlopen(req)
            logger.debug(res.code)  # 200  
            http_response=res.read().decode('utf-8')  #''<html>\n\t<title> you crack it</title>\n</html>\n''
            # If we need to monitor for an injected response, check that here
            if 'you crack it' in http_response:
                fitness += 60
            #if injected_http and injected_http in res.code:
            #    fitness -= 90
            #else:
            #    fitness += 100
        # except urllib.error.URLError as exc:
        #     logger.debug(exc)
        #     logger.exception(exc.reason)
        #     fitness += -101
            
        except requests.exceptions.ConnectTimeout as exc:
            logger.debug(exc)
            logger.exception("Socket timeout.[Fitness-100]")
            fitness -= 100
        except (requests.exceptions.ConnectionError, ConnectionResetError) as exc:
            logger.debug(exc)
            logger.exception("Connection RST.[Fitness-90]")
            fitness -= 90
        except urllib.error.HTTPError as exc:
            if(exc.code == 404):
                fitness += 50
        except urllib.error.URLError as exc:
            logger.debug(exc)
            logger.exception("HTTP Error [Fitness-101]")
            fitness -= 101   
        # Timeouts generally mean the strategy killed the TCP stream.
        # HTTPError usually mean the request was destroyed.
        # Punish this more harshly than getting caught by the censor.
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as exc:
            logger.debug(exc)
            logger.exception("[Fitness-120]")
            fitness += -120
        except Exception as exc:
            logger.debug(exc)
            logger.exception("Exception caught in HTTP test to site %s. [Fitness-100]", url)
            fitness += -100
        """
        logger.debug("Change Bitmap name with environment_id")
        bitmap_name='/mnt/hgfs/share-folders/bitmap/'+ args.get('environment_id')
        with open(bitmap_name,'wb') as f:
            f.write(open('/mnt/hgfs/share-folders/fuzz_bitmap','rb').read())
        """
        """
        while True:
            with open('/mnt/hgfs/share-folders/lock','r') as f:
                lock=f.read(1)
                if lock=='0':
                    break
               
        #在这里做一些fitness和代码覆盖率的操作
        logger.debug("Do Something with Bitmap")

        bitmap_name='/mnt/hgfs/share-folders/bitmap/'+'bitmap'+str(random.randint(0,100))
        with open(bitmap_name,'wb') as f:
            f.write(open('/mnt/hgfs/share-folders/fuzz_bitmap','rb').read())
    

        with open('/mnt/hgfs/share-folders/lock', 'w') as f:
            f.write('2\n')
        """

        return fitness * 4
