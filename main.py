import os

print("选择操作:")
print("1. 预处理数据")
print("2. 训练 AI")
print("3. 生成谱面")

choice = input("请输入选项 (1/2/3): ")

if choice == "1":
    os.system("python src/preprocess.py")
elif choice == "2":
    os.system("python src/train.py")
elif choice == "3":
    os.system("python src/infer.py")
else:
    print("无效选项")
