# encoding: utf-8
from __future__ import print_function
import os
import logging


def init_log(output_dir):
	logging.basicConfig(level=logging.INFO,
					format='%(asctime)s %(message)s',
					datefmt='%Y%m%d-%H:%M:%S',
					filename=os.path.join(output_dir, 'log.log'),
					filemode='w')
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	logging.getLogger('').addHandler(console)
	return logging

def close_log():
	handlers = logging.getLogger('').handlers[:]
	for handler in handlers:
		logging.getLogger('').removeHandler(handler)
		handler.close()

if __name__ == '__main__':
	logging = init_log("../test/")
	_print = logging.info
	_print('Train Epoch: {}/{} ...'.format(1, 2))
