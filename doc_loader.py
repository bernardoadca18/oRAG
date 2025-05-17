import csv

DOC_PATH = "./docs/"
documents = []

def load_csv(filename : str) -> list:
    """
    Load a CSV file and return its contents as a list of dictionaries.
    Each dictionary represents a row in the CSV file, with the keys being the column headers.
    """
    
    doc_list = []
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            i = 1
            for row in reader:
                clear_row = {
                    "id": f"{i}",
                    "text": str({
                            "name": row.get("name", ""),
                            "description": row.get("description", ""),
                            "price": row.get("price", "")
                        }),
                }
                i += 1
                if clear_row:
                    doc_list.append(clear_row)
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred while reading the file {filename}: {e}")
    return doc_list

documents = load_csv(DOC_PATH + "FILE.csv")

print("Loaded documents:")
for doc in documents:
    print(doc)