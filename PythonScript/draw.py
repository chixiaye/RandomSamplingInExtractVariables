import matplotlib.pyplot as plt

# Get input data from console
data = []
while True:
    try:
        input_str = input().strip()
        if input_str == "":
            break
        label, value = input_str.split(",")
        data.append((label.strip(), int(value.strip())))
    except ValueError:
        print("Invalid input format. Please input as 'label, value' separated by a comma.")
        continue

# Separate labels and values
labels = [item[0] for item in data]
values = [item[1] for item in data]

# Create pie chart
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

# Set title
ax.set_title("Pie Chart Example")

# Show plot
plt.show()
