import os, sys,random


def change_file_name(path, prefx, type):
	"""
	把文件下的所有文件重新按照序号命名
	:return:
	"""
	for index, filename in enumerate(os.listdir(path)):
		# shotname, extension = os.path.splitext(filename)
		if prefx:
			os.rename(os.path.join(path, filename), os.path.join(path, "%s_%d" % (prefx, index) + "." + type))
		else:
			os.rename(os.path.join(path, filename), os.path.join(path, str(index) + "." + type))


if __name__ == '__main__':
	# path = sys.argv[1]
	path = r"D:\Anjos\work\beijing_bank\code\ocr_pdf_to_json\pdf"
	# prefx = sys.argv[2]
	prefx = ""
	# type = sys.argv[3]
	type = "pdf"
	change_file_name(path, prefx, type)

