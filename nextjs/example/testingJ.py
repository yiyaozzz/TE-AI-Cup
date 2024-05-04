import json
import sys


def search_flags(data, path=''):
    flag_locations = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}/{key}" if path else key
            flag_locations.extend(search_flags(value, new_path))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = f"{path}[{index}]"
            if isinstance(item, str) and '_flag' in item:
                flag_locations.append(new_path)
            elif isinstance(item, (list, dict)):
                flag_locations.extend(search_flags(item, new_path))
    return flag_locations


def main():
    pdf_path = sys.argv[1]
    with open(pdf_path, 'r') as file:
        data = json.load(file)

    flag_locations = search_flags(data)

    # Iterate through each flag location, waiting for user input to continue
    for location in flag_locations:
        print(location)
        break


if __name__ == "__main__":
    main()
