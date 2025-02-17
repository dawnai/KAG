import hanlp
 # 语种见名称最后一个字段或相应语料库
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
tok_out=tok(['深圳南山今日（8日）發生巴士撞上巴士站，事後釀成2死1傷。深圳公安通報事件。通報指出，2月8日10時許，在南山區沙河西路茶光村公交站，一公交車入站停靠時因司機突發疾病與站台發生碰撞，造成3名候車乘客受傷，其中2人經搶救無效死亡。經對公交車司機進行呼氣式酒精測試，結果為0mg/100ml。目前，事故正在進一步調查處理'])
ner_out=ner(tok_out)
print(type(ner_out))
# ner = HanLP['ner']
# ner.dict_whitelist = {'司機': 'PERSON'}
# ner.dict_blacklist = {'1','2','3','4','5','6','7','8','9','一'}
# output=HanLP(['深圳南山今日（8日）發生巴士撞上巴士站，事後釀成2死1傷。深圳公安通報事件。通報指出，2月8日10時許，在南山區沙河西路茶光村公交站，一公交車入站停靠時因司機突發疾病與站台發生碰撞，造成3名候車乘客受傷，其中2人經搶救無效死亡。經對公交車司機進行呼氣式酒精測試，結果為0mg/100ml。目前，事故正在進一步調查處理'], tasks='ner')
# # print("抽取实体个数",len(output["ner/msra"][0]))
# print(output)
# import hanlp
# HanLP = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
# print(HanLP(['In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.',
#              '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
#              '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。']))