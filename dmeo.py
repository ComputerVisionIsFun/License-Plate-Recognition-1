import cv2






def demo_video(i_path, o_path, o_frame_rate, o_frame_width, o_frame_height):
    cap = cv2.VideoCapture(i_path)
    out = cv2.VideoWriter(o_path,cv2.VideoWriter_fourcc('M','J','P','G'), o_frame_rate, (o_frame_width,o_frame_height))

    # check 
    if (cap.isOpened()==False):
        print('Error opening video stream or file')

    # 
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:


            # processing

            # out.write()
            if cv2.waitKey(25)& 0xFF==ord('q'):
                break

        else:
            break

    cap.release()
    out.release()


    cv2.destroyAllWindows()


