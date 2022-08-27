import re
# with open("伤亡.txt","r",encoding="utf-8") as f:
#     all_text = f.read().split("\n")
#
# for text in all_text:
#     # result = re.findall(p,text) # 查找
#     text = re.sub("\s","",text)
#     text = re.sub(r"-*\d+\.?\d*%*","",text)
#     text = re.sub(r"[#\(\),、]","",text)
#
#     print(text)

# with open("Times.txt",encoding="utf-8") as f:
#     all_text = f.read()
#
# p = r"(\d{4})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})"
# p = r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}"
# r = re.findall(p,all_text)
# all_text = re.sub(p,r"\1年\2月\3日 \4时\5分\6秒",all_text)
# print(all_text)


text= "2019年3月1日"
# r = re.findall("(\d+)年",text)
r = re.search("(\d+)年",text)
print(r)