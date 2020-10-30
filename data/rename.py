import os


images_list = list(filter(lambda x: "s3" in x, os.listdir(".")))

for image_name in images_list:
    os.rename(image_name, image_name.replace("_s3", ""))
