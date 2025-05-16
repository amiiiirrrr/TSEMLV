
import pandas as pd

def calculate_mae(ground_truth_path, list_results):
    list_predicted = []
    list_type_predictions = []
    list_gt = []
    list_sorted_dicts = []
    # list_results: [{'image_name': "AGRMNPRZ_P2_11635", "width": " ", "hight": " ", "diagonal": " "}, {}, ...]

    # Load the Excel file
    excel_data = pd.read_excel(ground_truth_path, header=None, engine='openpyxl')

    # Iterate through rows
    for index, row in excel_data.iterrows():
        # print(row.values)
        if index > 0: # to pass the first row
            if row.values[0]=='Mean size':
                frame_numbers = row.values[5]
                # print("frame_numbers", frame_numbers)
                video_name = row.values[2]
                video_name = video_name.replace(" ", "")
                video_name = video_name.rstrip('\xa0')
                size_gt = row.values[4]
                type_measuring = row.values[6]
                frame_numbers = str(frame_numbers).split(',')
                for frame_number in frame_numbers:
                    frame_number = frame_number.replace(" ", "")
                    image_name_to_find = video_name + '_' + frame_number
                    
                    founded_dict = find_dictionary(image_name_to_find, list_results)
                    if founded_dict is None:
                        print("image_name_to_find", image_name_to_find)
                        continue
                    list_sorted_dicts.append(founded_dict)
                    list_gt.append(size_gt)
                    list_predicted.append(founded_dict[type_measuring])
                    list_type_predictions.append(type_measuring)

                    # print("image_name_to_find", image_name_to_find)
                    # print("list_gt", size_gt)
                    # print("list_predicted", founded_dict[type_measuring])

    MAE = mae(list_gt, list_predicted, list_type_predictions, list_sorted_dicts)
    return MAE

def find_dictionary(image_name_to_find, dictionary_list):
    '''
    find a specific dictionary in a list of dictionaries using one of the keys
    '''
    for dictionary in dictionary_list:
        if dictionary.get('image_name') == image_name_to_find:
            return dictionary
    return None  # Return None if not found

def mae(list1, list2, list_type_predictions, list_sorted_dicts):
    # Check if the input lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length.")

    # Convert elements to floats (if they are strings)
    list1 = [float(item) if isinstance(item, str) else item for item in list1]
    list2 = [float(item) if isinstance(item, str) else item for item in list2]
    list2 = ["{:.2f}".format(number) for number in list2]
    list2 = [float(item) for item in list2]

    # print("list_sorted_dicts", list_sorted_dicts)
    # print("list gt", len(list1))
    # print("list gt", sum(list1))
    # print("list predictions", len(list2))
    # print("list_type_predictions", list_type_predictions)
    # Calculate the absolute differences between corresponding elements

    ######################## To find the big errors #############################
    big_errors = []
    absolute_errors = []
    for a, b, dict_ in zip(list1, list2, list_sorted_dicts):
        error = abs(a - b)
        absolute_errors.append(error)
        if error > 5:
            big_errors.append(dict_['image_name'])
    # print("len(big_errors)", len(big_errors))
    # print("big_errors", big_errors)
    #######################################################################################

    ##################################################################################
    small_errors_big_tumors = []
    # absolute_errors = []
    for a, b, dict_ in zip(list1, list2, list_sorted_dicts):
        error = abs(a - b)
        # absolute_errors.append(error)
        if a > 20:
            # print('here')
            if error < 5:
                small_errors_big_tumors.append(dict_['image_name'])
    # print("len(small_errors_big_tumors)", len(small_errors_big_tumors))
    # print("small_errors_big_tumors", small_errors_big_tumors)
    #######################################################################################

    # Calculate the mean absolute error (MAE)
    # print("absolute_errors", absolute_errors)
    mae = sum(absolute_errors) / len(list1)

    # # to visualize
    # for iii in range(len(list1)):
    #     print("sorted_dict", list_sorted_dicts[iii])
    #     print("gt", list1[iii])
    #     print("prediction", list2[iii])
    #     print("type_prediction", list_type_predictions[iii])

    return mae