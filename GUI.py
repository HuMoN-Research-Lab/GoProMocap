import tkinter
import csv
import numpy as np
class firstGUI():
    '''Class for the GUI to create all project folders'''
    def __init__(self,root):
        self.root = root
    def createProject(self):
        with open('ProjectFoldersConfig.csv', newline='') as f:
            reader = csv.reader(f)#Read in the current CSV which has the config Variaables from the last time the script was run
            self.data = list(reader)#Put the variables in the list
        
        self.data = self.data[0] #Access the data (For some reason there is an extra empty dimension)
        newdata = [] #Create an empty list for the new user input
        
        #Using tkinter to create an Entry box for subject intials
        initials =tkinter.Label(self.root,text="Enter Subject Intials: ")
        self.initialEntry=tkinter.Entry(self.root)
        initials.grid(row = 0, column = 0)
        self.initialEntry.grid(row = 0, column = 1)
        self.initialEntry.focus_set()
        self.initialEntry.insert(0,self.data[0])
        
        #Using tkinter to create an Entry box for the date
        date =tkinter.Label(self.root,text="Enter date of recording: ")
        self.dateEntry=tkinter.Entry(self.root)
        date.grid(row = 1, column = 0)
        self.dateEntry.grid(row = 1, column = 1)
        self.dateEntry.focus_set()
        self.dateEntry.insert(0,self.data[1])

        #Using tkinter to create an Entry box for teh projct name
        projectname =tkinter.Label(self.root,text="Enter project name: ")
        self.projectEntry=tkinter.Entry(self.root)
        projectname.grid(row = 2, column = 0)
        self.projectEntry.grid(row = 2, column = 1)
        self.projectEntry.focus_set()
        self.projectEntry.insert(0,self.data[2])
        
        #Using tkinter to create an Entry box for the session number
        session =tkinter.Label(self.root,text="Enter session number: ")
        self.sessionEntry=tkinter.Entry(self.root)
        session.grid(row = 3, column = 0)
        self.sessionEntry.grid(row = 3, column = 1)
        self.sessionEntry.focus_set()
        self.sessionEntry.insert(0,self.data[3])
        
        #Using tkinter to create an Entry box for the where to save the project to
        baseProject =tkinter.Label(self.root,text="Enter the base folder path to save the project to: ")
        self.baseProjectEntry=tkinter.Entry(self.root)
        baseProject.grid(row = 4, column = 0)
        self.baseProjectEntry.grid(row = 4, column = 1)
        self.baseProjectEntry.focus_set()                                
        self.baseProjectEntry.insert(0,self.data[4])
        
        #Using tkinter to create an Entry box for the DeepLabCut config path
        DLCconfigPath =tkinter.Label(self.root,text="Enter DLC Config Path: ")
        self.DLCConfigEntry=tkinter.Entry(self.root)
        DLCconfigPath.grid(row = 0, column = 3)
        self.DLCConfigEntry.grid(row =0, column = 4, columnspan=2)
        self.DLCConfigEntry.focus_set() 
        self.DLCConfigEntry.insert(0,self.data[5])


        #Using tkinter to create an Entry box for the OpenPose file path
        OpenPosePath = tkinter.Label(self.root,text="Enter Path to Openpose Folder")
        self.OpenPosePathEntry=tkinter.Entry(self.root)
        OpenPosePath.grid(row= 1, column = 3)
        self.OpenPosePathEntry.grid(row = 1, column = 4, columnspan= 2)
        self.OpenPosePathEntry.focus_set()
        self.OpenPosePathEntry.insert(0,self.data[9])

        ##Using tkinter to create an Checkbutton for chessboard videos 
        var1 = tkinter.IntVar()
        var1.set(self.data[6])
        useCheckerBoardVid = tkinter.Checkbutton(self.root, text='Do you need to use chessboard videos?', var=var1) 
        self.useCheckerBoardVideEntry = tkinter.Entry(self.root)
        useCheckerBoardVid.grid(row = 3, column = 3)
        self.useCheckerBoardVideEntry.focus_set()
        
        ##Using tkinter to create an Checkbutton for calibrating Cameras
        var2 = tkinter.IntVar()
        var2.set(self.data[7])
        calibrateCameras = tkinter.Checkbutton(self.root, text='Do you need to calibrate cameras?', var=var2) 
        self.calibrateCamerasEntry = tkinter.Entry(self.root)
        calibrateCameras.grid(row = 4, column = 3)
        self.calibrateCamerasEntry.focus_set()
        '''
        ##Using tkinter to create an Checkbutton for portrait mode
        var3 = tkinter.IntVar()
        var3.set(self.data[8])
        portraitMode = tkinter.Checkbutton(self.root, text='Did you record the videos in portrait mode', var=var3) 
        self.portraitModeEntry = tkinter.Entry(self.root)
        portraitMode.grid(row = 5, column = 3)
        self.portraitModeEntry.focus_set()'''
        
        #Using tkinter to create an Entry box for the amount of cameras used
        numCams =tkinter.Label(self.root,text="Enter the number of cameras used in recording: ")
        self.numCams=tkinter.Entry(self.root)
        numCams.grid(row = 2, column = 3)
        self.numCams.grid(row =2, column = 4)
        self.numCams.focus_set() 
        self.numCams.insert(0,self.data[8])
        
        def buttonPush():
            '''Function for what is executed when the button is pushed'''
            #Append the newdata list with all the data that was inputted
            newdata.append(self.initialEntry.get())  
            newdata.append(self.dateEntry.get())
            newdata.append(self.projectEntry.get())
            newdata.append(self.sessionEntry.get())
            newdata.append(self.baseProjectEntry.get())
            newdata.append(self.DLCConfigEntry.get())
            newdata.append(var1.get())
            newdata.append(var2.get())
            newdata.append(self.numCams.get())
            newdata.append(self.OpenPosePathEntry.get())
            with open('ProjectFoldersConfig.csv', 'w') as f:#Open the csv file that stores the project info
                write = csv.writer(open('ProjectFoldersConfig.csv','w'), dialect = 'excel')
                write.writerow(newdata)#OverWrite the old config variables with new variables
            
            self.root.destroy()#Close GUI

        #Create a button to be pressed    
        runButton= tkinter.Button(self.root,text="Create Project Folders",
                           command=buttonPush).grid()
        
class secondGUI():
    '''Class for the GUI that has more specific details and runs the reconstruction code'''
    def __init__(self,root2):
        self.root2 =root2
    def runReconstruction(self):
        with open('runProject.csv', newline='') as f:
            reader = csv.reader(f) #Open the csv with the second GUI info
            self.data = list(reader)#And put those variables into a list
        with open('ProjectFoldersConfig.csv',newline='') as f:
            read = csv.reader(f)
            self.camNums = list(read)
        self.camNums = self.camNums[0]
        self.camNums = int(self.camNums[8])
        self.data = self.data[0]#Access data
        newdata = []#Create an empty list for new data

        instructions = tkinter.Text(self.root2, height=2, width = 130)
        instructions.grid(row=0, column = 0,  columnspan =6)
        instructions.insert(tkinter.END, 'Project Folders have been created in the specified folder path. Place videos into the raw video folder then enter the following   information. ')

        #Using tkinter to create an Entry box for camera ones name
        camera1 =tkinter.Label(self.root2,text="Enter camera one name: ")
        self.camera1Entry=tkinter.Entry(self.root2)
        camera1.grid(row = 1, column = 0)
        self.camera1Entry.grid(row = 1, column = 1)
        self.camera1Entry.focus_set()       
        self.camera1Entry.insert(0,self.data[0])
        
        
        #Using tkinter to create an Entry box foor camera twos name
        if self.camNums > 1:
            camera2 =tkinter.Label(self.root2,text="Enter camera two name: ")
            self.camera2Entry=tkinter.Entry(self.root2)
            camera2.grid(row = 2, column = 0)
            self.camera2Entry.grid(row = 2, column = 1)
            self.camera2Entry.focus_set() 
            self.camera2Entry.insert(0,self.data[1])
            
        #Using tkinter to create an Entry box for camera threes name
        if self.camNums > 2:
            camera3 =tkinter.Label(self.root2,text="Enter camera three name: ")
            self.camera3Entry=tkinter.Entry(self.root2)
            camera3.grid(row = 3, column = 0)
            self.camera3Entry.grid(row = 3, column = 1)
            self.camera3Entry.focus_set() 
            self.camera3Entry.insert(0,self.data[2])
            
        #Using tkinter to create an Entry box for camera fours name
        if self.camNums > 3:
            camera4 =tkinter.Label(self.root2,text="Enter camera four name: ")
            self.camera4Entry=tkinter.Entry(self.root2)
            camera4.grid(row = 4, column = 0)
            self.camera4Entry.grid(row = 4, column = 1)
            self.camera4Entry.focus_set() 
            self.camera4Entry.insert(0,self.data[3])
            
        #Using tkinter to create an Entry box for the base camera 
        baseCam = tkinter.Label(self.root2, text="Enter the name of the base Camera:")
        self.baseCamEntry = tkinter.Entry(self.root2)
        baseCam.grid(row = 1, column = 2)
        self.baseCamEntry.grid(row = 1, column =3)
        self.baseCamEntry.focus_set()
        self.baseCamEntry.insert(0,self.data[4])

        #Using tkinter to create an Entry box for the strat frame of reconstruction
        startFrame = tkinter.Label(self.root2, text="Enter the frame of video you want to start reconstruction:")
        self.startFrameEntry = tkinter.Entry(self.root2)
        startFrame.grid(row = 2, column = 2)
        self.startFrameEntry.grid(row = 2, column =3)
        self.startFrameEntry.focus_set()
        self.startFrameEntry.insert(0,self.data[5])

        #Using tkinter to create an Entry box for how long reconstrcution will be
        lenFrame = tkinter.Label(self.root2, text="Enter amount of frames you want to reconstruct (For full video enter -1):")
        self.lenFrameEntry = tkinter.Entry(self.root2)
        lenFrame.grid(row = 3, column = 2)
        self.lenFrameEntry.grid(row = 3, column =3)
        self.lenFrameEntry.focus_set()
        self.lenFrameEntry.insert(0,self.data[6])
        
        #Using tkinter to create an Entry box for rotating video
        rotateVid = tkinter.Label(self.root2, text='Enter if you need to rotate video. Enter ccw for counterclockwise, cw for clockwise or 0 for no rotate')
        self.rotateVidEntry = tkinter.Entry(self.root2)
        rotateVid.grid(row = 4, column =2)
        self.rotateVidEntry.grid(row =4, column = 3)
        self.rotateVidEntry.focus_set()
        self.rotateVidEntry.insert(0,self.data[11])

        ##Using tkinter to create a checkbutton for including deeplabcut
        var1 = tkinter.IntVar()
        var1.set(self.data[7])
        include_DLC = tkinter.Checkbutton(self.root2, text='Do you want include DeepLabCut?', var=var1) 
        self.includeDLCEntry = tkinter.Entry(self.root2)
        include_DLC.grid(row = 1, column = 5)
        self.includeDLCEntry.focus_set()

        ##Using tkinter to create a checkbutton for including Openpose Face
        var2 = tkinter.IntVar()
        var2.set(self.data[8])
        include_OpenPoseFace = tkinter.Checkbutton(self.root2, text='Do you want include OpenPose Face Points?', var=var2) 
        self.include_OpenPoseFaceEntry = tkinter.Entry(self.root2)
        include_OpenPoseFace.grid(row = 2, column = 5)
        self.include_OpenPoseFaceEntry.focus_set()

        ##Using tkinter to create a checkbutton for including openpose hand points
        var3 = tkinter.IntVar()
        var3.set(self.data[9])
        include_OpenPoseHands = tkinter.Checkbutton(self.root2, text='Do you want include OpenPose Hand Points?', var=var3) 
        self.include_OpenPoseHandsEntry = tkinter.Entry(self.root2)
        include_OpenPoseHands.grid(row = 3, column = 5)
        self.include_OpenPoseHandsEntry.focus_set()

        ##Using tkinter to create a checkbutton for including openpose skeleton
        var4 = tkinter.IntVar()
        var4.set(self.data[10])
        include_OpenPoseSkeleton = tkinter.Checkbutton(self.root2, text='Do you want include OpenPose Body Points?', var=var4) 
        self.include_OpenPoseSkeletonEntry = tkinter.Entry(self.root2)
        include_OpenPoseSkeleton.grid(row = 4, column = 5)
        self.include_OpenPoseSkeletonEntry.focus_set()
        
        def buttonPush():
            '''Function for what is executed when button is pushed'''
            #Append the newdata list with all data from the User inputs            
            newdata.append(self.camera1Entry.get())
            if self.camNums > 1:  
                newdata.append(self.camera2Entry.get())
            else:
                newdata.append('0')
            if self.camNums >2:
                newdata.append(self.camera3Entry.get())
            else:
                newdata.append('0')
            if self.camNums>3:
                newdata.append(self.camera4Entry.get())
            else:
                newdata.append('0')
            newdata.append(self.baseCamEntry.get())
            newdata.append(self.startFrameEntry.get())
            newdata.append(self.lenFrameEntry.get())
            newdata.append(var1.get())
            newdata.append(var2.get())
            newdata.append(var3.get())
            newdata.append(var4.get())
            newdata.append(self.rotateVidEntry.get())

            with open('runProject.csv', 'w') as f:#Open the csv file that stores the variables
                write = csv.writer(open('runProject.csv','w'), dialect = 'excel')
                write.writerow(newdata)#Overwrite the data with new data
            
            self.root2.destroy()#Close th GUI

        #Create a button     
        runButton = tkinter.Button(self.root2,text="Run Reconstruction",
                           command=buttonPush).grid()
        
        
        
        
        

