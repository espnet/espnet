from xlrd import open_workbook
import pickle


f1 = 'pinyin.xlsx'
f2 = 'taigi.xlsx'

phone_dict = {str(i): i for i in range(1, 11)}

with open_workbook(f1) as wb:
    sheet = wb.sheets()[0]

    for row_index in range(2, sheet.nrows):
        row = sheet.row_values(row_index)
        ipa_diphtong = row[3]
        ls = ipa_diphtong.split('-')
        for t in ls:
            if t in phone_dict:
                continue
            else:
                phone_dict[t] = len(phone_dict) + 1


with open_workbook(f2) as wb:
    sheet = wb.sheets()[0]

    for row_index in range(1, sheet.nrows):
        row = sheet.row_values(row_index)
        ipa_diphtong = row[3]
        ls = ipa_diphtong.split('-')
        for t in ls:
            if t in phone_dict:
                continue
            else:
                phone_dict[t] = len(phone_dict) + 1

phone_dict[","] = len(phone_dict)+1
phone_dict["."] = len(phone_dict)+1
output = open('phone_dict.pkl', 'wb')
print(len(phone_dict))
pickle.dump(phone_dict, output)
