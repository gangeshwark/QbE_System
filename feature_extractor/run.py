import subprocess, inspect
import os
from os.path import basename

import feature_extractor.readArk
from feature_extractor.data_preparation import prepare_dir


class FeatureExtractor():
    @staticmethod
    def bnf(file_path):
        os.chdir(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
        # extract file name from path
        file_name = basename(file_path)
        file_name = os.path.splitext(file_name)[0]
        subprocess.call("./path.sh", shell=True)
        print "Place the 'hierbn-nd-train80k' folder in your home directory (~/)"
        nnet_dir = os.path.abspath(os.getenv("HOME")+'/hierbn-nd-train80k')
        nj = 1
        audio = file_path
        output_dir = os.path.abspath('../outdir/data/%s' % (file_name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prepare_dir(audio, output_dir)

        test_fbank_pitch_feat = '../outdir/fbank_pitch_%s' % (file_name)  # physical location of fbank feats
        test_bnf_feat = '../outdir/bnf_%s' % (file_name)  # physical location of bnf feats
        path = "./feat_extr_test.sh %s %s %s %s %s false" % (
            output_dir, nj, test_fbank_pitch_feat, test_bnf_feat, nnet_dir)
        subprocess.call(path, shell=True)
        return os.path.abspath(test_bnf_feat + '/raw_bnfea_fbank_pitch.1.scp')


if __name__ == '__main__':
    print FeatureExtractor.bnf('/home/gangeshwark/PycharmProjects/QbE_System/uploads/hello1.wav')
