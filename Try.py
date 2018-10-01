import openpyxl

kitap = openpyxl.load_workbook('RALtoRGB.xlsx')
sayfa = kitap.get_sheet_by_name('Sayfa1')
# sayfa.append(["RAL 7001"])

list = []
for i in sayfa['A']:
    list.append(str(i.value))

print(list)
kitap.save('RALtoRGB.xlsx')
kitap.close()
