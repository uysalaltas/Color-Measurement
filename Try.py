import openpyxl

r_code = []
g_code = []
b_code = []
RAL = []

doc = openpyxl.load_workbook('RALtoRGB.xlsx')
paper = doc.get_sheet_by_name('Sayfa1')

for i in paper['A']:
    RAL.append(str(i.value))

for r, g, b in zip(paper['B'], paper['C'], paper['D']):
    r_code.append(r.value)
    g_code.append(g.value)
    b_code.append(b.value)

doc.save('RALtoRGB.xlsx')
doc.close()
