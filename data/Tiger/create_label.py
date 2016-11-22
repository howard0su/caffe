import json
import os.path
import random
from PIL import Image

width = 640
fp = open('training/label.idl', 'r')
trainval = open('trainval.txt', 'w')
valid = open('test.txt', 'w')

s = fp.readline()
while s != '':
	result = json.loads(s)
	name = result.keys()[0]
	id = os.path.splitext(os.path.basename(name))[0]
	lablefn = 'label/' + id + '.txt'
	txtfile = open(lablefn, 'w')	
	for entry in result[name] :
		if entry[4] == 20:
			entry[4] = 4
		txtfile.write ('{4} {0} {1} {2} {3}\n'.format(entry[0], entry[1], entry[2], entry[3], entry[4]))
	txtfile.close()

	# 10% data used for validation
	if random.randint(0, 9) == 5:	
		valid.write ('{0} {1}\n'.format('training/'+name, lablefn))
	else:
		trainval.write ('{0} {1}\n'.format('training/'+name, lablefn))
		img = Image.open('training/' + name)
		img.transpose(Image.FLIP_LEFT_RIGHT)
		imagefn = 'gen/' + id + '_.jpg';
		img.save(imagefn, "JPEG")
		lable2fn = 'label/' + id + '_.txt'
		txtfile = open(lable2fn, 'w')
		for entry in result[name] :
			txtfile.write ('{4} {0} {1} {2} {3}\n'.format(width - entry[2], entry[1], width - entry[0], entry[3], entry[4]))
		txtfile.close()
		trainval.write ('{0} {1}\n'.format(imagefn, lable2fn))
	s = fp.readline()
trainval.close()
valid.close()
fp.close()
