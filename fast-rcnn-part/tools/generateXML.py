import xml.dom.minidom
import shutil
import os
CLASSES=('__background__','aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
  
def GenerateXml(imagename,width_data,height_data,score,Labels,BBox):    
  impl = xml.dom.minidom.getDOMImplementation()  
  dom = impl.createDocument(None, 'annotation', None)  
  root = dom.documentElement    
  folder =  dom.createElement('folder')  
  File =  dom.createElement('filename')
  Size =  dom.createElement('size') 
 # Object = dom.createElement('object')  
  root.appendChild(folder)
  root.appendChild(File)
  root.appendChild(Size)
  #root.appendChild(Object)  
    
  foldername=dom.createTextNode('VOC2007')
  filename=dom.createTextNode(imagename)
  
  folder.appendChild(foldername)
  File.appendChild(filename)
 
  width=dom.createElement('width')
  height=dom.createElement('height')
  depth=dom.createElement('depth')
   
  width_str=str(width_data)
  height_str=str(height_data)

  widthnum=dom.createTextNode(width_str)
  heightnum=dom.createTextNode(height_str)
  depthnum=dom.createTextNode('3')
 
  width.appendChild(widthnum)
  height.appendChild(heightnum)
  depth.appendChild(depthnum)
  Size.appendChild(width)
  Size.appendChild(height)
  Size.appendChild(depth)

  for i in xrange(len(Labels)):
       Object = dom.createElement('object')
       root.appendChild(Object) 
      
       nameE=dom.createElement('name')  
       nameT=dom.createTextNode(CLASSES[Labels[i]])  
       nameE.appendChild(nameT) 
       
       Score= dom.createElement('score')
       score_num=dom.createTextNode(str(score[i]))
       Score.appendChild(score_num)

       box=dom.createElement('bndbox') 
       xmin=dom.createElement('xmin') 
       ymin=dom.createElement('ymin')
       xmax=dom.createElement('xmax')
       ymax=dom.createElement('ymax')
 
       xminnum=dom.createTextNode(str(BBox[i*4+0]))
       yminnum=dom.createTextNode(str(BBox[i*4+1]))
       xmaxnum=dom.createTextNode(str(BBox[i*4+2]))   
       ymaxnum=dom.createTextNode(str(BBox[i*4+3]))
  
       xmin.appendChild(xminnum)
       ymin.appendChild(yminnum)
       xmax.appendChild(xmaxnum)
       ymax.appendChild(ymaxnum)
       box.appendChild(xmin)
       box.appendChild(ymin)
       box.appendChild(xmax)
       box.appendChild(ymax)
  
       difficult=dom.createElement('difficult')
       diff_num=dom.createTextNode('0')
       difficult.appendChild(diff_num)
    
       Object.appendChild(nameE)  
       Object.appendChild(Score)
       Object.appendChild(box)
       Object.appendChild(difficult)
  folder_annotation="data/VOCdevkit20/VOC20/Predictions/"
  prexml=imagename.split(".")[0]
  f= open(folder_annotation+prexml+'.xml', 'w')  
  dom.writexml(f, addindent='  ', newl='\n')  
  f.close()       
