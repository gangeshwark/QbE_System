import codecs
import os
import random
import sys
import contextlib
import wave

def prepare_dir(testFilesStr, datadir):
    file1 = codecs.open(datadir + '/utt2spk','w','utf-8')
    file2 = codecs.open(datadir + '/spk2utt','w','utf-8')
    file3 = codecs.open(datadir + '/text','w','utf-8')
    file4 = codecs.open(datadir + '/segments','w','utf-8')
    file5 = codecs.open(datadir + '/wav.scp','w','utf-8')

    testFiles = testFilesStr.split(';')
    for i in range(len(testFiles)):
        name = os.path.splitext(os.path.basename(testFiles[i]))[0]
        with contextlib.closing(wave.open(testFiles[i],'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        file1.write("spk0" + str(i+1) + "_" + name + " " + "spk0" + str(i+1) + "\n")
        file2.write("spk0" + str(i+1) + " " + "spk0" + str(i+1) + "_" + name + "\n")
        file3.write("spk0" + str(i+1) + "_" + name + " text" + "\n")
        file4.write("spk0" + str(i+1) + "_" + name + " " + name + " 0 " + str(duration) + "\n")
        file5.write(name + " /usr/bin/sox " + testFiles[i] + " -r 8000 -c 1 -b 16 -t wav - |\n")
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    

if __name__ == '__main__':
    if(len(sys.argv)<3):
        print("USAGE: python " + sys.argv[0] + " testFilesStr datadir")
        exit(1)
    testFilesStr = sys.argv[1]  
    datadir = sys.argv[2]
    print ("testFilesStr = " + testFilesStr + ", datadir = " + datadir)
    prepare_dir(testFilesStr, datadir)

