import cv2
import random
import argparse
import os
import json
import warnings

lower_red = (0, 100, 100)
upper_red = (10, 255, 255)

def addLabel(image, label):
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_image, lower_red, upper_red)
  non_zero_pixels = cv2.findNonZero(mask)
  if non_zero_pixels is None:
        raise Exception("No non-zero pixels found in the mask")

  try:
      temp = non_zero_pixels.reshape(-1, 2).tolist()
  except Exception as e:
      print(f"Error in finding non zero: {e}")
      raise

  red_pixels = temp
  random.shuffle(red_pixels)

  flag = False
  for x, y in red_pixels:
    # Check if rectangle is completely red
    if is_fully_red(image, x, y, 30):
      # Define rectangle center
      bl_x = int(x)
      bl_y = int(y + 30)

      # Write "1" on the image (adjust font size and color as needed)
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 1
      font_thickness = 2
      text_color = (0, 0, 0)  # White text
      cv2.putText(image, text=str(label), org=(bl_x, bl_y), fontFace=font, fontScale=font_scale, color=text_color, thickness=font_thickness)
      flag = True
      break

  if not flag:
    raise Exception("Could not find space for label")


def is_fully_red(image, x, y, size):
  if x < 0 or y < 0 or x + size >= image.shape[1] or y + size >= image.shape[0]:
    return False

  hsv_image = cv2.cvtColor(image[y:y+size, x:x+size], cv2.COLOR_BGR2HSV)

  return cv2.countNonZero(cv2.inRange(hsv_image, lower_red, upper_red)) == size * size


def addBackground(image, bg_path, img_size=(1024, 1024)):
    if bg_path is None or bg_path == "":
        return image
    
    bg = cv2.imread(bg_path)
    height, width = bg.shape[:2]
    left = (width - height) // 2
    top = 0
    right = (width + height) // 2
    bottom = height
    bg = bg[top:bottom, left:right]
    bg = cv2.resize(bg, img_size)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(image, image, mask=mask_inv)
    bg = cv2.bitwise_and(bg, bg, mask=mask)
    final = cv2.add(fg, bg)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default="new_depth_images/visible")
    parser.add_argument('-l', '--put_bg', action='store_true')
    parser.add_argument('-b', '--bg_dir', type=str, default="../real_world_bg")
    parser.add_argument('-i', '--id', action='store_true')
    parser.add_argument('-r', '--id_reverse', action='store_true')
    args = parser.parse_args()

    path = args.path
    truth_path = f"{path}/../truths"
    img_path = f"{path}/../imgs"

    if args.id and args.id_reverse:
        print("Cannot have both id and id_reverse")
        exit(1)

    if args.id:
        os.makedirs(f"{path}/../labelled_id", exist_ok=True)
    elif args.id_reverse:
        os.makedirs(f"{path}/../labelled_reverse_id", exist_ok=True)
    else:
        os.makedirs(f"{path}/../labelled", exist_ok=True)


    print("Creating Images")

    # for i in [99, 269, 321, 341]:
    for i in range(len(os.listdir(img_path))):
        if os.path.exists(f"{path}/img{i}_shape0.jpg") is False:
            print(f"Could not find image {i}") 
            continue
        
        finalImage = cv2.imread(f"{path}/img{i}_shape0.jpg")
        truth = json.load(open(f"{truth_path}/truth{i}.json"))

        j = 0
        try:
            if args.id:
                addLabel(finalImage, truth['shapes'][j]['id'])
            elif args.id_reverse:
                addLabel(finalImage, truth['shapes'][len(truth['shapes']) - j - 1]['id'])
            else:
                addLabel(finalImage, truth['shapes'][j]['label'])
        except Exception as e:
            print(f"Error in image {i} shape {j}: {e}")
            i += 1
            continue
        flag = False
        while True:
            j += 1
            if not os.path.exists(f"{path}/img{i}_shape{j}.jpg"):
                break

            nextImage = cv2.imread(f"{path}/img{i}_shape{j}.jpg")
            try:
                if args.id:
                    addLabel(nextImage, truth['shapes'][j]['id'])
                elif args.id_reverse:
                    addLabel(nextImage, truth['shapes'][len(truth['shapes']) - j - 1]['id'])
                else:
                    addLabel(nextImage, truth['shapes'][j]['label'])
            except Exception as e:
                print(f"Error in image {i} shape {j}: {e}")
                flag = True
                break
            finalImage = cv2.bitwise_not(cv2.add(cv2.bitwise_not(finalImage), cv2.bitwise_not(nextImage)))

        if flag:
            i += 1
            continue
        
        if args.put_bg:
            bg_path = f"{args.bg_dir}/{truth['bg']}"
        else:
            bg_path = None

        finalImage = addBackground(finalImage, bg_path, finalImage.shape[:2])

        if args.id:
            cv2.imwrite(f"{path}/../labelled_id/img{i}_labelled.jpg", finalImage)
        elif args.id_reverse:
            cv2.imwrite(f"{path}/../labelled_reverse_id/img{i}_labelled.jpg", finalImage)
        else:
            cv2.imwrite(f"{path}/../labelled/img{i}_labelled.jpg", finalImage)
        i+=1
          
        