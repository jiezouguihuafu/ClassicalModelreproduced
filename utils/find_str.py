import re

# -*- coding: UTF-8 -*-

"""判断一个unicode是否是汉字"""
def is_chinese(uchar):
	if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
		return True
	else:
		return False

"""判断一个unicode是否是数字"""
def is_number(uchar):
	if uchar >= u'\u0030' and uchar <= u'\u0039':
		return True
	else:
		return False

"""判断一个unicode是否是英文字母"""
def is_alphabet(uchar):
	if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
		return True
	else:
		return False


"""判断是否是（汉字，数字和英文字符之外的）其他字符"""
def is_other(uchar):
	if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
		return True
	else:
		return False

"""半角转全角"""
def B2Q(uchar):
	inside_code = ord(uchar)
	if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
		return uchar
	if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
		inside_code = 0x3000
	else:
		inside_code += 0xfee0
	return unichr(inside_code)


"""全角转半角"""
def Q2B(uchar):
	inside_code = ord(uchar)
	if inside_code == 0x3000:
		inside_code = 0x0020
	else:
		inside_code -= 0xfee0
	if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
		return uchar
	return unichr(inside_code)

"""把字符串全角转半角"""
def stringQ2B(ustring):
	return "".join([Q2B(uchar) for uchar in ustring])

"""将UTF-8编码转换为Unicode编码"""
def convert_toUnicode(string):
	ustring = string
	if not isinstance(string, unicode):
		ustring = string.decode('UTF-8')
	return ustring

# 检测是否包含杂乱字符
def if_contain_chaos(keyword):
	try:
		keyword.encode("gb2312")
	except UnicodeEncodeError:
		return True
	return False

if __name__ == "__main__":

	ustring1 = '收割季节 麦浪和月光 洗着快镰刀'
	string1 = 'Sky0天地Earth1*'

	print(if_contain_chaos("  第 @20@00l0 号验资报告。 截至 20@4年 l2 月 26 日止， 变更后的累计注册资本为人民币 375，398，@屁·00元 (大写: 人民币叁亿柒仟伍佰叁拾玖万捌仟壹佰柒拾贰元整九"))




