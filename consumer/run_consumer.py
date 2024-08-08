from Consumer import Consumer



if __name__ == "__main__":
    import cv2
    import random

    consumer = Consumer()
    consumer()




    # cap = cv2.VideoCapture(r'C:\Users\ASUS\Desktop\github_projects\Parking\parking-management\main_vid.mp4')

    # data = {}
    # counter = 0
    # while True:
    #     ret, frame = cap.read()
        
    #     if not ret:
    #         print("Reached the end of the video.")
    #         break
            
    #     frame, info = consumer(frame)    
        

    #     cv2.imwrite(rf"C:\Users\ASUS\Desktop\github_projects\Parking\parking-management\temp\{str(counter).zfill(4)}.png", frame)


    #     counter += 1
        
    # cap.release()
    # cv2.destroyAllWindows()

