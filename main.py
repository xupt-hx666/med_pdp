import os
from server import FedPer


def main():
    if os.path.exists('pneumonia_model.pth'):
        print("==== 发现训练好的模型，启动肺炎检测界面 ====")
        from pneumonia_gui import main as gui_main
        gui_main()
    else:
        print("==== 未发现训练模型，开始联邦学习训练 ====")
        fed_system = FedPer()
        fed_system.run()
        from pneumonia_gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()

"python main.py"

