
def show_metrics(method, name: str, iter: int, save: bool = False) -> None:
    total_dict_all = {}
    dict_all_ant = {}

    for i in range(iter):
        list_all = []
        print("Iteración: ", i+1)

        met = method(save)

        for ele in met:
            list_all = met[ele]
            if (dict_all_ant == {}):
                total_dict_all[ele] = list_all
            else:
                total_dict_all[ele] = list(
                    map(lambda x, y: x+y, dict_all_ant[ele], list_all))
        dict_all_ant = total_dict_all

    write_metrics(total_dict_all, name, iter)

# Pointer networks


def show_metrics_load(method, name: str, iter: int) -> None:
    total_dict_all = {}
    list_all_ant = []

    for i in range(iter):
        list_all = []
        print("Iteración: ", i+1)
        met = method(name)
        list_all = met
        if (list_all_ant == []):
            total_dict_all[name] = list_all
        else:
            total_dict_all[name] = list(
                map(lambda x, y: x+y, list_all_ant, list_all))
        list_all_ant = total_dict_all[name]

    write_metrics(total_dict_all, name, iter)


def write_metrics(total_list_all: dict, name: str, iter: int) -> None:
    f = open("./results/medidas_totales.txt", "a")
    for ele in total_list_all:
        list_all_ele = total_list_all[ele]
        f.write(f"{name} para {iter} iteraciones con el modelo {ele}\n")
        f.write(f"Accuracy Test_x: {list_all_ele[0]/iter}\n")
        f.write(f"Time: {list_all_ele[1]/iter}\n")
        f.write(f"Recall: {list_all_ele[2]/iter}\n")
        f.write(f"Precision: {list_all_ele[3]/iter}\n")
        f.write(f"F1: {list_all_ele[4]/iter}\n")
        f.write(f"Cross Validation: {list_all_ele[5]/iter}\n")
        f.write(f"STD: {list_all_ele[6]/iter}\n")
        f.write(
            "-------------------------------------------------------------------------\n")
    f.close()
