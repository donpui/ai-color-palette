from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def calculate_saliency(image, saliency_map, colors):
    # Convert saliency map to binary
    _, saliency_binary = cv2.threshold(saliency_map, 127, 255, cv2.THRESH_BINARY)

    saliency_counts = {}
    
    for color in colors:
        # Create binary mask for current color
        lower = np.array([color[0] - 10, color[1] - 10, color[2] - 10])
        upper = np.array([color[0] + 10, color[1] + 10, color[2] + 10])
        mask = cv2.inRange(image, lower, upper)

        # Count white pixels where color mask and saliency map overlap
        saliency_counts[RGB2HEX(color)] = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=saliency_binary))

    labels = list(saliency_counts.keys())
    sizes = list(saliency_counts.values())

    # print labels with sizes
    for i in range(len(labels)):
        print(labels[i], sizes[i])
        

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

def get_colors_top(image, number_of_colors, show_chart=True):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    
    # sort clusters by their counts and get the top 5
    counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    top_counts = {k: counts[k] for k in list(counts)[:5]}
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in top_counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in range(len(top_counts.keys()))]
    rgb_colors = [ordered_colors[i] for i in range(len(top_counts.keys()))]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        
        # calculate the percentages
        #total = sum(top_counts.values())
        #percentages = [(count / total) * 100 for count in top_counts.values()]

        plt.pie(top_counts.values(), labels = hex_colors, colors = hex_colors, autopct='%1.1f%%')
        plt.show() # This keeps the plot window open
    
    return rgb_colors


def get_colors_percentage(image, number_of_colors, show_chart=True):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        
        # calculate the percentages
        total = sum(counts.values())
        percentages = [(count / total) * 100 for count in counts.values()]

        chart = plt.pie(counts.values(), labels = hex_colors, colors = hex_colors, autopct='%1.1f%%')
        plt.show() # This keeps the plot window open
    
    return rgb_colors

def get_colors(image, number_of_colors, show_chart=True):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        plt.show()    

    return rgb_colors

# Replace 'image.jpg' with your image path
#get_colors_top(get_image('image.png'), 8, True)

image = get_image('original.jpg')
saliency_map = cv2.cvtColor(cv2.imread('saliency.jpg'), cv2.COLOR_BGR2GRAY)

colors = get_colors_top(image, 5, show_chart=True)[:5]

# Calculate saliency for 8 dominant colors
calculate_saliency(image, saliency_map, colors)

