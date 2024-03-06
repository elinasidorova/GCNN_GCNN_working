from data_manager import DataManager


if __name__ == "__main__":
    element = input("Select a chemical element to add data for: ")
    file_name = f"../Data/csv/{element}.csv"
    data = DataManager(file_name)

    # molecules adding loop
    while True:
        warning = ''

        try:
            new_data = data.from_keyboard()
        except KeyboardInterrupt:
            break

        if new_data['name'][0] in data.get_molecules():
            warning = input(f"{new_data['name'][0]} already exists. Add anyway? (y/n): ")

        if warning == 'n':
            print('Aborted! NEW ENTRY:\n')
            continue

        data.add_data(new_data)
        print("\nNEW ENTRY:\n")

    print(f'\n{data.get_counter()} molecules was successfully added!\nTotal: {data.get_amount()}')
    data.save_data()


