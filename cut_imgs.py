from PIL import Image

# 打开图像
image = Image.open('bengio.png')

# 获取图像的宽度和高度
image_width, image_height = image.size  # 1132 x 906

# hinton 1 4

# 定义行和列
rows, cols = 4, 5

# 计算每个子图的宽度和高度
sub_image_width = image_width // cols  # 1132 // 5 = 226
sub_image_height = image_height // rows  # 906 // 4 = 226

# 定义你想要提取的图片的索引（行和列，从0开始计数）
row_index = 3  # 第三行
col_index = 0  # 第四列

# 计算裁剪区域 (left, upper, right, lower)
left = col_index * sub_image_width
upper = row_index * sub_image_height
right = left + sub_image_width
lower = upper + sub_image_height

# 裁剪该区域
cropped_image = image.crop((left, upper, right, lower))

# 保存或展示裁剪后的图片
cropped_image.save('./celebrity/bengio.png')
cropped_image.show()
