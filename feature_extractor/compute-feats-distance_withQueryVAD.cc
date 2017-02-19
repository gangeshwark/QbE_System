#include "base/io-funcs.h"
#include "util/kaldi-io.h"
#include "base/kaldi-math.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"
#include "mgb-lib.h"
#include <string>

using namespace std;
struct Detection {
	string file;
	float score;
	double segStart;
	double segEnd;
	int end;
};



namespace kaldi {
struct Pos {
	BaseFloat score;
	int pos;
};

struct less_than_key
{
    inline bool operator() (const Pos& struct1, const Pos& struct2)
    {
        return (struct1.score < struct2.score);
    }
};

void MakeSubMatrixVector(const Matrix<BaseFloat> &mat_utt,
                         const std::string &utt_name,
                         const float frame_length,
                         const std::string &boundary_name, 
                         RandomAccessBaseFloatMatrixReader &file_reader, 
                         std::vector<Matrix<BaseFloat>* > *vec_mat,
                         std::vector<std::pair<BaseFloat, BaseFloat> > *vec_time_bound) {
  if(!boundary_name.empty() && !file_reader.HasKey(utt_name))
    KALDI_ERR << "ERROR, MakeSubMatrixVector: no boundary info for utterance: " 
                    << "'" << utt_name << "'";
  if(boundary_name.empty()) {
    Matrix<BaseFloat> *mat = new Matrix<BaseFloat>;
    *mat = mat_utt;
    vec_mat->push_back (mat);
    if(vec_time_bound != NULL) {
      BaseFloat sec_start = 0;
      BaseFloat sec_end = static_cast<BaseFloat> (mat_utt.NumRows() * frame_length);
      vec_time_bound->push_back(std::pair<BaseFloat, BaseFloat>(sec_start, sec_end));
    }
    return;
  }
  int32 frame_num = mat_utt.NumRows();
  Matrix<BaseFloat> mat_boundary = file_reader.Value(utt_name);
  KALDI_LOG << utt_name << ": " << mat_boundary.NumRows() << " subsequences";
  for(int32 i = 0; i < mat_boundary.NumRows(); i++) {
    BaseFloat sec_start = mat_boundary(i, 0);
    int32 frame_start = static_cast<int32> (sec_start/frame_length);
    KALDI_ASSERT(0 <= frame_start && frame_start < frame_num);
    BaseFloat sec_end = mat_boundary(i, 1);
    int32 frame_end = static_cast<int32> (sec_end/frame_length);
    if (vec_time_bound != NULL) {
      vec_time_bound->push_back(std::pair<BaseFloat, BaseFloat>(sec_start, sec_end));
    }
    if (!(0 <= frame_end && frame_end < frame_num && frame_start < frame_end)) {
      KALDI_WARN << "frame_start=" << frame_start << ' '
                << "frame_end=" << frame_end << ' ' 
                << "frame_num=" << frame_num;
      frame_end = frame_num;
    }
    SubMatrix<BaseFloat> max_sub(mat_utt, frame_start, frame_end - frame_start, 0, mat_utt.NumCols());
    Matrix<BaseFloat> *mat = new Matrix<BaseFloat>;
    *mat = max_sub;
    vec_mat->push_back(mat);
  }
}


void MakeSubMatrixVector_query(const Matrix<BaseFloat> &mat_utt,
                         const std::string &utt_name,
                         const float frame_length,
                         std::vector<Matrix<BaseFloat>* > *vec_mat,
                         std::map<string, pair<int, int> > boundMap) {
    std::pair<int, int> bound = boundMap.find(utt_name)->second;
	int start = bound.first;
	int end = bound.second;
	cout << "Utter = " << utt_name  << ", start = " << start << ", end = " << end << endl;

	SubMatrix<BaseFloat> max_sub(mat_utt, start, end - start, 0, mat_utt.NumCols());
    Matrix<BaseFloat> *mat = new Matrix<BaseFloat>;
    *mat = max_sub;
    vec_mat->push_back(mat);
}



void getTimeBound(string timeBoundFile, std::map<string, pair<int, int> > &boundMap)
{
	SequentialBaseFloatMatrixReader query_bound_reader;
	query_bound_reader.Open(timeBoundFile);
	for(; !query_bound_reader.Done(); query_bound_reader.Next()) {
		string utt = query_bound_reader.Key();
		Matrix<BaseFloat> mat_utt1 = query_bound_reader.Value();
		cout << "Key:" << utt << endl;
		int startFrame = 0, endFrame = mat_utt1.NumCols() - 1;
		while(startFrame < endFrame) {
			bool isBound = true;
			for(int i = 0; i < 5;i++)
				if(mat_utt1(0,startFrame + i) == 0)isBound = false;
			if(isBound)break;
			startFrame++;
		}
		while(endFrame >= startFrame) {
			bool isBound = true;
			for(int i = 0; i < 5;i++)
				if(mat_utt1(0,endFrame - i) == 0)isBound = false;
			if(isBound)break;
			endFrame--;
		}
		if(startFrame < 0) startFrame = 0;
		if(endFrame > mat_utt1.NumCols() - 1)endFrame = mat_utt1.NumCols() - 1;
		pair<int, int> p(startFrame, endFrame);
		boundMap[utt] = p;
	}
}
 
//   
void FreeSubMatrixVector(std::vector<Matrix<BaseFloat>* > *vec_mat) {
  for(size_t i = 0; i < vec_mat->size(); i++) {
    Matrix<BaseFloat> *mat = (*vec_mat)[i];
    delete mat;
  }
  vec_mat->resize(0);
}
//
void ComputeDist(const CuMatrix<BaseFloat> &mat1, const CuMatrix<BaseFloat> &mat2, 
                 CuMatrix<BaseFloat> *mat_dist) {
  KALDI_ASSERT(mat1.NumCols() == mat2.NumCols());
  mat_dist->Resize(mat1.NumRows(), mat2.NumRows());
  CuMatrix<BaseFloat> mat1_t = mat1;
  mat1_t.Transpose();
  CuMatrix<BaseFloat> mat2_t = mat2;
  mat2_t.Transpose();
  
  mat_dist->AddMatMat(1.0, mat1, kNoTrans, mat2_t, kNoTrans, 0);
  CuMatrix<BaseFloat> mat_scale(mat1.NumRows(), mat2.NumRows());
  for(int32 i = 0; i < mat1.NumRows(); i ++) {
    CuVector<BaseFloat> v1(mat1.NumCols());
    v1.CopyColFromMat(mat1_t, i);
    BaseFloat x = v1.Norm(2);
    for(int32 j = 0; j < mat2.NumRows(); j ++) {
      CuVector<BaseFloat> v2(mat2.NumCols());
      v2.CopyColFromMat(mat2_t, j);
      BaseFloat y = v2.Norm(2);
      mat_scale(i, j) = 1.0/(x * y); 
    }
  }
  mat_dist->MulElements(mat_scale);  // get the cosine distance score
  CuMatrix<BaseFloat> mat_unit(mat1.NumRows(), mat2.NumRows());
  mat_unit.Set(1.0);
  mat_dist->Scale(-1.0);
  mat_dist->AddMat(1.0, mat_unit);  // get the final score 1 - distance_score
}

void findListPos(std::vector<Pos> &listScore, std::vector<BaseFloat> &minCost, std::vector<int> &endPos, BaseFloat min, int queryLength)
{
	std::sort(listScore.begin(), listScore.end(), less_than_key());
	std::cout << "\nScore list after sorting (first 1000 scores):";
  	for(int i = 0; i < listScore.size();i++)
	{
		std::cout << listScore[i].pos << "," << listScore[i].score << "  ";
		if(i >= 1000) break;
	}
  	std::cout << endl; 
	
	int numSelected = 0;
	int n = listScore.size();
	int* selected = NULL;
	selected = new int[n];
	for(int i = 0; i < n;i++)
		selected[i] = 0;
	for(int i = 0; i < n;i++)
	{
		if(numSelected >= 50) break;
		//if(listScore[i].score >= min * 1.1) break;
		if(selected[listScore[i].pos] == 1) continue;
		minCost.push_back(listScore[i].score);
		endPos.push_back(listScore[i].pos);
		numSelected++;
		int startMarked = listScore[i].pos - queryLength;
		if(startMarked < 0) startMarked = 0;
		int endMarked = listScore[i].pos + queryLength;
		if(endMarked >= n) endMarked = n-1;
		for(int j = startMarked; j <= endMarked;j++)
			selected[j] = 1;
	}
	delete [] selected; selected = NULL;	
}

//
void GetMinCost(const BaseFloat diag_length_penalty, 
                const Matrix<BaseFloat> &mat_dist, std::vector<BaseFloat> &minCost, std::vector<int> &endPos, BaseFloatMatrixWriter &feats_writer, string matName) {
  int32 num_rows = mat_dist.NumRows();
  int32 num_cols = mat_dist.NumCols();
  Matrix<BaseFloat> match_curve_len(num_rows, num_cols);
  Matrix<BaseFloat> total_cost(num_rows, num_cols);
  Matrix<BaseFloat> aver_cost (num_rows, num_cols);
  
  for(int32 i = 0; i < num_cols; i++) {  // first column initialization, horizontal
    total_cost(0, i) = mat_dist(0, i);
    match_curve_len(0, i) = 1.0;
    aver_cost(0, i) = total_cost(0, i);
  }
  for(int32 j = 1; j < num_rows; j ++) { // row initialization  vertical
    total_cost(j, 0) = total_cost(j-1, 0) + mat_dist(j, 0);
    match_curve_len(j, 0) = match_curve_len(j-1, 0) + 1;
    aver_cost(j, 0) = total_cost(j, 0)/match_curve_len(j, 0);  
  }

  for(int32 i = 1; i < num_rows; i ++) {
    for(int32 j = 1; j < num_cols; j ++) {
      BaseFloat cur_dist = mat_dist(i, j);
      BaseFloat cost_h = total_cost(i, j-1) + cur_dist;
      BaseFloat cost_v = total_cost(i-1, j) + cur_dist;
      BaseFloat cost_d = total_cost(i-1, j-1) + cur_dist;
      
      BaseFloat curve_len_h = match_curve_len(i, j-1);
      BaseFloat curve_len_v = match_curve_len(i-1, j);
      BaseFloat curve_len_d = match_curve_len(i-1, j-1);

      BaseFloat aver_cost_h = cost_h / (1 + curve_len_h);
      BaseFloat aver_cost_v = cost_v / (1 + curve_len_v);
      BaseFloat aver_cost_d = cost_d / (diag_length_penalty + curve_len_d);
      if (aver_cost_h < aver_cost_v) {
        if(aver_cost_h < aver_cost_d) {
          total_cost(i, j) = cost_h;
          aver_cost(i, j) = aver_cost_h;
          match_curve_len(i, j) = 1 + curve_len_h;
        } else {
          total_cost(i, j) = cost_d;
          aver_cost(i, j) = aver_cost_d;
          match_curve_len(i, j) = diag_length_penalty + curve_len_d;
        }
      } else if (aver_cost_v < aver_cost_d) {
        total_cost(i, j) = cost_v;
        aver_cost(i, j) = aver_cost_v;
        match_curve_len(i, j) = 1 + curve_len_v;
      } else {
        total_cost(i, j) = cost_d;
        aver_cost(i, j) = aver_cost_d;
        match_curve_len(i, j) = diag_length_penalty + curve_len_d;
      }
    }
  }
  feats_writer.Write(matName + "-distMatt", mat_dist);
  feats_writer.Write(matName + "-totalCost", total_cost);
  feats_writer.Write(matName + "-averCost", aver_cost);
  feats_writer.Write(matName + "-curveLen", match_curve_len);
  BaseFloat min = aver_cost(num_rows-1, 0);
  //endPos = 1;
  std::vector<Pos> listScore;
  Pos first;
  first.score = aver_cost(num_rows-1, 0);
  first.pos = 0;
  listScore.push_back(first);
  for(int32 i = 1; i < num_cols; i ++)  {
    if(aver_cost(num_rows - 1, i) < min)
	{
      min = aver_cost(num_rows - 1, i);
	  //endPos = i + 1;
	}
	Pos p;
	p.score = aver_cost(num_rows - 1, i);
	p.pos = i;
	listScore.push_back(p);
  }

  std::cout << "Min = " << min << ", query length = " << num_rows << ", audio length = " << num_cols << ", number of scores = " << listScore.size() << ", score list (First 1000 scores):" << endl;
  for(int i = 0; i < listScore.size();i++)
  {
	std::cout << aver_cost(num_rows - 1, i) << " ";
	if(i >= 1000) break;
  }
  std::cout << endl; 
  
  findListPos(listScore, minCost, endPos, min, num_rows);
  
}
//
void ComputeFeatDistances(const BaseFloat diag_length_penalty, 
                          std::vector<Matrix<BaseFloat>* > &vec_mat1, 
                          std::vector<Matrix<BaseFloat>* > &vec_mat2,
                          std::vector<Detection> &listDet, BaseFloat threshold, std::vector<std::pair<BaseFloat, BaseFloat> > vec_time_bound, BaseFloatMatrixWriter &feats_writer, string matName) {
  for(int32 i = 0; i < vec_mat1.size(); i++) {
    for(int32 j = 0; j < vec_mat2.size(); j++) {
      CuMatrix<BaseFloat> cu_mat_dist;
      ComputeDist(CuMatrix<BaseFloat>(*vec_mat1[i]), CuMatrix<BaseFloat>(*vec_mat2[j]), &cu_mat_dist);
      Matrix<BaseFloat> mat_dist(cu_mat_dist.NumRows(), cu_mat_dist.NumCols());
      cu_mat_dist.CopyToMat(&mat_dist);
      std::vector<BaseFloat> curMinCost;
      std::vector<int> endPos; 
	  string istr = static_cast<ostringstream*>( &(ostringstream() << i) )->str();
	  string jstr = static_cast<ostringstream*>( &(ostringstream() << j) )->str();
	  string newMatName = matName + "_" + istr + "_" + jstr;
	  GetMinCost(diag_length_penalty, mat_dist, curMinCost, endPos, feats_writer, newMatName);
	  
      //KALDI_ASSERT(curMinCost <= 1.0);
	  for(int k = 0; k < endPos.size();k++)
	  {
			cout << "Detection " << (k+1) << ": pos = " << endPos[k] << ", score = " << curMinCost[k] << endl;
        	Detection d;
			d.score = curMinCost[k];
			d.segStart = vec_time_bound[j].first;
			d.segEnd = vec_time_bound[j].second;
			d.end = endPos[k];
			listDet.push_back(d);
		
      }     
    }
  }
}

}  // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
    "Compute the minimum distance between two feature streams(matrices) using DTW method.\n"
    "compute-feats-distance [options] scp:feats1.scp scp:feats2.scp dist.txt\n";
    ParseOptions po(usage);
    float frame_length=0.01;
    std::string src2tgt = "";
    std::string src_time_boundary_file = "", tgt_time_boundary_file = "";
    po.Register("frame-length", &frame_length, "frame duration");
    float diag_length_penalty = 2.0;
    po.Register("diag-length-penalty", &diag_length_penalty, "diag length penalty");
    po.Register("src2tgt", &src2tgt, "source to target name mapping file");
    po.Register("src-time-boundary-file", &src_time_boundary_file, "time boundary info for the source files");
    po.Register("tgt-time-boundary-file", &tgt_time_boundary_file, "time boundary info for the target files");
    std::string use_gpu = "no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    po.Read(argc, argv);
    if(po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }
    std::string feature1_rspecifier = po.GetArg(1);
    std::string feature2_rspecifier = po.GetArg(2);
    std::string dist_wfilename = po.GetArg(3);
	std::string trackback_w = po.GetArg(4);
	std::string boundary_query = po.GetArg(5); 

	std::map<string, pair<int, int> > boundMap;
	getTimeBound(boundary_query, boundMap);
	std::cout << "Welcome" << std::endl;
#if HAVE_CUDA==1
  CuDevice::Instantiate().SelectGpuId(use_gpu);
  CuDevice::Instantiate().DisableCaching();
#endif    
    SequentialBaseFloatMatrixReader feature1_reader(feature1_rspecifier);
    SequentialBaseFloatMatrixReader feature2_reader(feature2_rspecifier);
    std::map<std::string, std::map<std::string, unsigned short> > map_src2tgt;
    StrMapInMap s2t_map;
    if (!src2tgt.empty()) {
      s2t_map.Open(src2tgt);
    }
    RandomAccessBaseFloatMatrixReader boundary1_reader, boundary2_reader;
    if(!src_time_boundary_file.empty()) {
      boundary1_reader.Open(src_time_boundary_file);
    }
    if(!tgt_time_boundary_file.empty()) {
      boundary2_reader.Open(tgt_time_boundary_file);
    }
    bool output_binary =false;
    Output output_kaldi(dist_wfilename, output_binary);
	BaseFloatMatrixWriter feats_writer(trackback_w);//e.g. "ark:traceback_matt.ark"
    for(; !feature1_reader.Done(); feature1_reader.Next()) {
      std::string utt1 = feature1_reader.Key();
      Matrix<BaseFloat> mat_utt1 = feature1_reader.Value();
      std::vector<Matrix<BaseFloat>*> vec_mat_utt1;

	  std::pair<int, int> bound = boundMap.find(utt1)->second;
	  int start = bound.first;
	  int end = bound.second;
	  float queryLength = (end-start + 1) * frame_length;
	  std::cout << "Start = " << start << ", end = " << end << ", query length = " << queryLength << std::endl;

      MakeSubMatrixVector_query(mat_utt1, utt1, frame_length, &vec_mat_utt1, boundMap); 
      for(; !feature2_reader.Done(); feature2_reader.Next()) {
        std::string utt2 = feature2_reader.Key();
        if(!src2tgt.empty() && !s2t_map.HasValue(utt1, utt2))
          continue;
        Matrix<BaseFloat> mat_utt2 = feature2_reader.Value();
        std::vector<Matrix<BaseFloat>*> vec_mat_utt2;
        std::vector<std::pair<BaseFloat, BaseFloat> > vec_time_bound;
        MakeSubMatrixVector(mat_utt2, utt2, frame_length, tgt_time_boundary_file, boundary2_reader, &vec_mat_utt2, &vec_time_bound);
        std::vector<Detection> listDet;
        //int32 minIndex; 
		string matName = utt1 + "_" + utt2;
        ComputeFeatDistances(diag_length_penalty, vec_mat_utt1, vec_mat_utt2, listDet, 10, vec_time_bound, feats_writer, matName);
		std::cout << "Number of detection = " << listDet.size() << std::endl;
		for(int i = 0; i < listDet.size();i++)
		{
			float startDet = listDet[i].end * frame_length + listDet[i].segStart - queryLength;
			if(startDet < 0) startDet = 0;
			output_kaldi.Stream() << utt2 << " " << listDet[i].segStart << " " << listDet[i].segEnd << " " << listDet[i].end << " " << startDet << " " << listDet[i].score << "\n";
		}
        //KALDI_ASSERT(minIndex < vec_time_bound.size());
        //output_kaldi.Stream() << utt1 << ' ' << utt2 << ' ' << vec_time_bound[minIndex].first 
        //                      << ' ' << vec_time_bound[minIndex].second << ' '
        //                      << minCost << "\n";
        FreeSubMatrixVector(&vec_mat_utt2);    
      }
      FreeSubMatrixVector(&vec_mat_utt1);
      feature2_reader.Close();  // is there any rewind function for this ?
      feature2_reader.Open(feature2_rspecifier);
    }
	feats_writer.Close();
    output_kaldi.Close(); 
  } catch (std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
  return 0;
}


