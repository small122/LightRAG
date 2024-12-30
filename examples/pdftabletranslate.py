import pdfplumber
import os
import re
import json

# 设置保存路径
txt_file_path = r"C:\Users\10519\Downloads\pdf_output.txt"
image_folder_path = r"C:\Users\10519\Downloads\pdf_images"

# 确保图像保存文件夹存在
os.makedirs(image_folder_path, exist_ok=True)

# 打开PDF文件
pdf_path = r"C:\Users\10519\Downloads\1-21：污水综合排放标准GB8978-1996.pdf"
pdf = pdfplumber.open(pdf_path)

# 创建或清空txt文件
with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    # 遍历PDF的每一页
    for page_num, page in enumerate(pdf.pages):
        # 提取页面的文本内容
        # text = page.extract_text()
        # if text:
        #     # 判断是否为图表部分
        #     # 使用特定的关键词或模式检测图表，例如"表格"、"图"、序号结构等
        #     # 清理页眉、页脚等无关信息
        #     cleaned_text = re.sub(r"(页\s*码|第\s*\d+页|版次|前言|目录).*", "", text)
        #
        #     # 写入清理后的文本
        #     txt_file.write(f"--- 第 {page_num + 1} 页文本 ---\n")
        #     txt_file.write(cleaned_text.strip())
        #     txt_file.write("\n\n")

        # 提取页面的表格
        tables = page.extract_tables()
        if tables:
            txt_file.write(f"--- 第 {page_num + 1} 页表格 ---\n")
            struct_data = []
            root = tables[0][0]
            tableroot = []
            name = ''
            number = 1
            for i in range(len(root)):
                if root[i] is not None:
                    name = root[i].replace("\n","")
                    tableroot.append(root[i])
                else:
                    root[i] = (str(name) + str(number)).replace("\n","")
                    tableroot.append(root[i])
                    number += 1
            for j in range(1, len(tables[0])):
                entry = {}
                for i in range(len(tableroot)):
                    if tables[0][j][i] is None:
                        tables[0][j][i] = tables[0][j - 1][i]
                    entry[tableroot[i]] = tables[0][j][i].replace("\n","")
                struct_data.append(entry)
            # 转换为JSON格式
            json_string = json.dumps(struct_data, ensure_ascii=False, indent=4)
            txt_file.write(json_string)
            txt_file.write("\n\n")

        # 将页面转换为图像并保存
        image = page.to_image(resolution=150)
        image_path = os.path.join(image_folder_path, f"page_{page_num + 1}.png")
        image.save(image_path)

# 关闭PDF文件
pdf.close()

print(f"PDF处理完成，结果已保存到 {txt_file_path} 和 {image_folder_path} 文件夹。")
