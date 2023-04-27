# Prompt the user to choose an option
print("Please choose an option for game recommendation:")
print("1. Collaborative Filtering")
print("2. Content-Based Filtering")
print("3. TF-IDF")

# Get the user's choice
choice = int(input("Enter your choice (1-3): "))

# Execute the chosen option
if choice == 1:
    print("Starting Collaborative Filtering..")
    # execute collaborative.py script
    exec(open("collaborative.py").read())
elif choice == 2:
    print("Starting Content-Based Filtering..")
    # execute content-based.py script
    exec(open("content-based.py").read())
elif choice == 3:
    print("Starting TF-IDF..")
    # execute td-idf.py script
    exec(open("tf-idf.py").read())
else:
    print("Invalid choice. Please enter a number between 1 and 3.")
