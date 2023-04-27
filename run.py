# Prompt the user to choose an option
print("Please choose an option for game recommendation:")
print("1. Collaborative Filtering")
print("2. Content-Based Filtering")
print("3. TF-IDF")

# Get the user's choice
choice = int(input("Enter your choice (1-3): "))

# Execute the chosen option
if choice == 1:
    print("You chose Option 1")
    # execute collaborative.py script
    exec(open("collaborative.py").read())
elif choice == 2:
    print("You chose Option 2")
    # execute content-based.py script
    exec(open("content-based.py").read())
elif choice == 3:
    print("You chose Option 3")
    # execute td-idf.py script
    exec(open("td-idf.py").read())
else:
    print("Invalid choice. Please enter a number between 1 and 3.")
