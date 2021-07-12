/*
 * @Email: yueshangChang@gmail.com
 * @Author: nanmi
 * @Date: 2021-06-29 15:58:00
 * @LastEditors: zerollzeng
 * @LastEditTime: 2020-05-22 11:49:13
 */

#include "Trt.h"
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include "opencv2/cudaarithm.hpp"

class InputParser{                                                              
    public:                                                                     
        InputParser (int &argc, char **argv){                                   
            for (int i=1; i < argc; ++i)                                        
                this->tokens.push_back(std::string(argv[i]));                   
        }                                                                       
        const std::string& getCmdOption(const std::string &option) const{       
            std::vector<std::string>::const_iterator itr;                       
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option); 
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){      
                return *itr;                                                    
            }                                                                   
            static const std::string empty_string("");                          
            return empty_string;                                                
        }                                                                       
        bool cmdOptionExists(const std::string &option) const{                  
            return std::find(this->tokens.begin(), this->tokens.end(), option)  
                   != this->tokens.end();                                       
        }                                                                       
    private:                                                                    
        std::vector <std::string> tokens;                                       
};  

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t--onnx\t\tinput onnx model, must specify\n"
              << "\t--batch_size\t\tdefault is 1\n"
              << "\t--mode\t\t0 for fp32 1 for fp16 2 for int8, default is 0\n"
              << "\t--engine\t\tsaved path for engine file, if path exists, "
                  "will load the engine file, otherwise will create the engine file "
                  "after build engine. dafault is empty\n"
              << "\t--calibrate_data\t\tdata path for calibrate data which contain "
                 "npz files, default is empty\n"
              << "\t--gpu\t\tchoose your device, default is 0\n"
              << "\t--dla\t\tset dla core if you want with 0,1..., default is -1(not enable)\n"
              << std::endl;
}

// image resize to target shape
struct Location
{
    int w;
    int h;
    int x;
    int y;
    cv::cuda::GpuMat Img;
};

Location ImageProcess(cv::Mat &srcImage, const int &th, const int &tw, const int &type)
{
    int w, h, x, y;
    float r_w = (float)(tw / (srcImage.cols*1.0));
    float r_h = (float)(th / (srcImage.rows*1.0));
    if (r_h > r_w) {
        w = tw;
        h = (int)(r_w * srcImage.rows);
        x = 0;
        y = (th - h) / 2;
    } else {
        w = (int)(r_h * srcImage.cols);
        h = th;
        x = (tw - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, type);
    cv::cuda::GpuMat input(srcImage);
    cv::cuda::GpuMat output(re);
    input.upload(srcImage);
    cv::cuda::resize(input, output, re.size());

    Location out{w, h, x, y, output};
    return out;
}

#define kCONF_THRESH 0.4
#define kNMS_THRESH 0.5

#define BATCH_SIZE 1
#define INPUT_H 320
#define INPUT_W 320

#define NUM_CLASS 80
int reg_max = 7;



struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};

struct BoxInfo 
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
};


const float kmean[3] = { 103.53, 116.28, 123.675 };
const float kstd[3] = { 57.375, 57.12, 58.395 };
// void decode_infer(float* cls_pred, float* dis_pred, int stride, float threshold, std::vector<std::vector<BoxInfo>>& results);


inline float fast_exp(float x) 
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) 
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}


void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}

BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;

        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)INPUT_H);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)INPUT_W);

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}

void decode_infer(const float* cls_pred, const float* dis_pred, const int& stride, const float& threshold, std::vector<std::vector<BoxInfo>>& results)
{
    int feature_h = INPUT_W / stride;
    int feature_w = INPUT_H / stride;

    for (int idx = 0; idx < feature_h * feature_w; idx++)
    {
        const float* scores = cls_pred + idx*80;

        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        float all = 0;
        for (int label = 0; label < NUM_CLASS; label++)
        {
            all += scores[label];
            if (scores[label] > score)
            {
                score = scores[label];
                cur_label = label;
            }
        }

        if (score > threshold)
        {
            const float* bbox_pred = dis_pred + idx*32;
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
        }
    }
}

void decode_infer(const cv::Mat cls_pred, const cv::Mat dis_pred, const int& stride, const float& threshold, std::vector<std::vector<BoxInfo>>& results)
{
    int feature_h = INPUT_W / stride;
    int feature_w = INPUT_H / stride;

    for (int idx = 0; idx < feature_h * feature_w; idx++)
    {
        const float* scores = cls_pred.ptr<float>(idx);

        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        auto max_pos = std::max_element(scores, scores + NUM_CLASS);
        score = *max_pos;
        cur_label = max_pos - scores;
        // for (int label = 0; label < NUM_CLASS; label++)
        // {
        //     all += scores[label];
        //     if (scores[label] > score)
        //     {
        //         score = scores[label];
        //         cur_label = label;
        //     }
        // }

        if (score > threshold)
        {
            const float* bbox_pred = dis_pred.ptr<float>(idx);
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
        }
    }
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}


// ====================================================

void softmax(float* x, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		x[i] = std::exp(x[i]);
		sum += x[i];
	}
	for (i = 0; i < length; i++)
	{
		x[i] /= sum;
	}
}



void generate_proposal(std::vector<int>& classIds, std::vector<float>& confidences, std::vector<cv::Rect>& boxes, const int stride_, cv::Mat out_score, cv::Mat out_box)
{
    const int num_grid_y = INPUT_H / stride_;
    const int num_grid_x = INPUT_W / stride_;
    const int reg_1max = reg_max + 1;

    if(out_score.dims==3)
    {
        out_score = out_score.reshape(0, num_grid_x*num_grid_y);
    }
    if(out_box.dims==3)
    {
        out_box = out_box.reshape(0, num_grid_x*num_grid_y);
    }
    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const int idx = i * num_grid_x + j;
			cv::Mat scores = out_score.row(idx).colRange(0, NUM_CLASS);
			cv::Point classIdPoint;
			double score;
			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, 0, &score, 0, &classIdPoint);
            std::cout << "::::::::: " << score << std::endl;
            if (score >= kCONF_THRESH)
            {
                float* pbox = (float*)out_box.data + idx * reg_1max * 4;
                float dis_pred[4];
                for (int k = 0; k < 4; k++)
                {
                    softmax(pbox, reg_1max);
                    float dis = 0.f;
                    for (int l = 0; l < reg_1max; l++)
                    {
                        dis += l * pbox[l];
                    }
                    dis_pred[k] = dis * stride_;
                    pbox += reg_1max;
                }

				float pb_cx = (j + 0.5f) * stride_ - 0.5;
                float pb_cy = (i + 0.5f) * stride_ - 0.5;       
                float x0 = pb_cx - dis_pred[0];
                float y0 = pb_cy - dis_pred[1];
                float x1 = pb_cx + dis_pred[2];
                float y1 = pb_cy + dis_pred[3];

                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
                boxes.push_back(cv::Rect((int)x0, (int)y0, (int)(x1 - x0), (int)(y1 - y0)));
            }
        }
    }
}

float* normalize(cv::Mat& img, const float* kmean, const float* kstd)
{
    // img.convertTo(img, CV_32F);
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i]                         = (float)((uc_pixel[2] - kmean[0]) / kstd[0]);
            data[i + INPUT_H * INPUT_W]     = (float)((uc_pixel[1] - kmean[1]) / kstd[1]);
            data[i + 2 * INPUT_H * INPUT_W] = (float)((uc_pixel[0] - kmean[2]) / kstd[2]);
            uc_pixel += 3;
            ++i;
        }
    }
    return data;
}

float* normalize(cv::Mat& img)
{
    // img.convertTo(img, CV_32F);
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = (float)(uc_pixel[2] / 255.0);
            data[i + INPUT_H * INPUT_W] = (float)(uc_pixel[1] / 255.0);
            data[i + 2 * INPUT_H * INPUT_W] = (float)(uc_pixel[0] / 255.0);
            uc_pixel += 3;
            ++i;
        }
    }
    return data;
}

const int color_list[80][3] =
{
    //{255 ,255 ,255}, //bg
    {216 , 82 , 24},
    {236 ,176 , 31},
    {125 , 46 ,141},
    {118 ,171 , 47},
    { 76 ,189 ,237},
    {238 , 19 , 46},
    { 76 , 76 , 76},
    {153 ,153 ,153},
    {255 ,  0 ,  0},
    {255 ,127 ,  0},
    {190 ,190 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 ,255},
    {170 ,  0 ,255},
    { 84 , 84 ,  0},
    { 84 ,170 ,  0},
    { 84 ,255 ,  0},
    {170 , 84 ,  0},
    {170 ,170 ,  0},
    {170 ,255 ,  0},
    {255 , 84 ,  0},
    {255 ,170 ,  0},
    {255 ,255 ,  0},
    {  0 , 84 ,127},
    {  0 ,170 ,127},
    {  0 ,255 ,127},
    { 84 ,  0 ,127},
    { 84 , 84 ,127},
    { 84 ,170 ,127},
    { 84 ,255 ,127},
    {170 ,  0 ,127},
    {170 , 84 ,127},
    {170 ,170 ,127},
    {170 ,255 ,127},
    {255 ,  0 ,127},
    {255 , 84 ,127},
    {255 ,170 ,127},
    {255 ,255 ,127},
    {  0 , 84 ,255},
    {  0 ,170 ,255},
    {  0 ,255 ,255},
    { 84 ,  0 ,255},
    { 84 , 84 ,255},
    { 84 ,170 ,255},
    { 84 ,255 ,255},
    {170 ,  0 ,255},
    {170 , 84 ,255},
    {170 ,170 ,255},
    {170 ,255 ,255},
    {255 ,  0 ,255},
    {255 , 84 ,255},
    {255 ,170 ,255},
    { 42 ,  0 ,  0},
    { 84 ,  0 ,  0},
    {127 ,  0 ,  0},
    {170 ,  0 ,  0},
    {212 ,  0 ,  0},
    {255 ,  0 ,  0},
    {  0 , 42 ,  0},
    {  0 , 84 ,  0},
    {  0 ,127 ,  0},
    {  0 ,170 ,  0},
    {  0 ,212 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 , 42},
    {  0 ,  0 , 84},
    {  0 ,  0 ,127},
    {  0 ,  0 ,170},
    {  0 ,  0 ,212},
    {  0 ,  0 ,255},
    {  0 ,  0 ,  0},
    { 36 , 36 , 36},
    { 72 , 72 , 72},
    {109 ,109 ,109},
    {145 ,145 ,145},
    {182 ,182 ,182},
    {218 ,218 ,218},
    {  0 ,113 ,188},
    { 80 ,182 ,188},
    {127 ,127 ,  0},
};

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

void draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic light", "fire hydrant",
                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                        "baseball glove", "skateboard", "surfboard", "tennis racket",
                                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;


    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio), 
                                      cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
    cv::imwrite("test_res.jpg", image);
}


int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    //std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        //std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        //std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    //cv::imshow("dst", dst);
    //cv::waitKey(0);
    return 0;
}


int main(int argc, char** argv) {
    
    // parse args
    if (argc < 2) {
        show_usage(argv[0]);
        return 1;
    }
    InputParser cmdparams(argc, argv);

    const std::string& onnx_path = cmdparams.getCmdOption("--onnx");
    
    std::vector<std::string> custom_outputs;
    const std::string& custom_outputs_string = cmdparams.getCmdOption("--custom_outputs");
    std::istringstream stream(custom_outputs_string);
    if(custom_outputs_string != "") {
        std::string s;
        while (std::getline(stream, s, ',')) {
            custom_outputs.push_back(s);
        }
    }

    int run_mode = 0;
    const std::string& run_mode_string = cmdparams.getCmdOption("--mode");
    if(run_mode_string != "") {
        run_mode = std::stoi(run_mode_string);
    }

    const std::string& engine_file = cmdparams.getCmdOption("--engine");

    int batch_size = 1;
    const std::string& batch_size_string = cmdparams.getCmdOption("--batch_size");
    if(batch_size_string != "") {
        batch_size = std::stoi(batch_size_string);
    }

    const std::string& calibrateDataDir = cmdparams.getCmdOption("--calibrate_data");
    const std::string& calibrateCache = cmdparams.getCmdOption("--calibrate_cache");

    int device = 0;
    const std::string& device_string = cmdparams.getCmdOption("--gpu");
    if(device_string != "") {
        device = std::stoi(device_string);
    }

    int dla_core = -1;
    const std::string& dla_core_string = cmdparams.getCmdOption("--dla");
    if(dla_core_string != "") {
        dla_core = std::stoi(dla_core_string);
    }
    
    // build engine
    Trt* onnx_net = new Trt();
    onnx_net->SetDevice(device);
    onnx_net->SetDLACore(dla_core);
    if(calibrateDataDir != "" || calibrateCache != "") {
        onnx_net->SetInt8Calibrator("Int8EntropyCalibrator2", batch_size, calibrateDataDir, calibrateCache);
    }
    onnx_net->CreateEngine(onnx_path, engine_file, custom_outputs, batch_size, run_mode);
    
    // input 
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

    cv::namedWindow("low light video enhanced");
    bool test_img = true;
    bool test_video = false;

    if (test_img)
    {
        const char* file_names = "../../image/test.jpg";
        cv::Mat img = cv::imread(file_names);
        if (img.empty()) std::cerr << "Read image failed!" << std::endl;

        auto time_start = std::chrono::steady_clock::now();
        int gpu = 0;
        int cpu = 1;
        if (gpu)
        {
            cv::Mat re;
            cv::cuda::GpuMat input;

            cv::cuda::GpuMat output_img;
            input.upload(img);

            cv::cuda::resize(input, output_img, cv::Size(INPUT_W, INPUT_H));
            
            cv::cuda::GpuMat flt_image(INPUT_H, INPUT_W, CV_32FC3);
            output_img.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
            
            std::vector<cv::cuda::GpuMat> chw;

            float* gpu_input = (float*)(onnx_net->mBinding[0]);

            for (size_t i = 0; i < 3; ++i)
            {
                chw.emplace_back(cv::cuda::GpuMat(cv::Size(INPUT_W, INPUT_H), CV_32FC1, gpu_input + i * INPUT_W * INPUT_H));
            }
            cv::cuda::split(flt_image, chw);

            onnx_net->CopyFromHostToDevice(gpu_input, 0);
            
            // do inference
            auto time1 = std::chrono::steady_clock::now();
            onnx_net->Forward();
            auto time2 = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
            std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;

            // trt output
            float* gpu_output = (float*)(onnx_net->mBinding[1]);
            
            cv::cuda::GpuMat flt_image_out;
            cv::cuda::GpuMat out_put;

            std::vector<cv::cuda::GpuMat> chw_1;
            for (size_t i = 0; i < 3; ++i)
            {
                chw_1.emplace_back(cv::cuda::GpuMat(cv::Size(INPUT_W, INPUT_H), CV_32FC1, gpu_output + i * INPUT_W * INPUT_H));
            }
            cv::cuda::merge(chw_1, out_put);

            cv::cuda::GpuMat image_out;
            out_put.convertTo(image_out, CV_32FC3, 1.f * 255.f);
            cv::cuda::resize(image_out, flt_image_out, img.size());
            
            cv::Mat dst;
            flt_image_out.download(dst);

            auto time_end = std::chrono::steady_clock::now();
            auto all_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
            std::cout << "Whole time: " << all_duration << " ms. TRT enqueue time: " << duration << " ms. FPS: " << (1000 / all_duration) << std::endl;

            cv::imwrite("../01_test_demo.jpg", dst);
            // cv::imshow("low light video enhanced", image_out.Img);
        }
        if (cpu)
        {
            cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
            cv::resize(img, out, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
            cv::Mat pr_img = out; // letterbox BGR to RGB


            object_rect effect_roi;
            cv::Mat resized_img;
            resize_uniform(img, resized_img, cv::Size(320, 320), effect_roi);



            // float* kinput = normalize(pr_img, kmean, kstd);
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = resized_img.data + row * resized_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[i]                         = (float)((uc_pixel[0] - kmean[0]) / kstd[0]);
                    data[i + INPUT_H * INPUT_W]     = (float)((uc_pixel[1] - kmean[1]) / kstd[1]);
                    data[i + 2 * INPUT_H * INPUT_W] = (float)((uc_pixel[2] - kmean[2]) / kstd[2]);
                    uc_pixel += 3;
                    ++i;
                }
            }
            std::vector<float> input(data, data + sizeof(data)/sizeof(float));
            onnx_net->CopyFromHostToDevice(input, 0);
            // do inference
            auto time1 = std::chrono::steady_clock::now();
            onnx_net->Forward();
            auto time2 = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
            std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;
            
            // get output
            // std::vector<float> output;
            // onnx_net->CopyFromDeviceToHost(output, 1);

            std::vector<float> dis_8;
            std::vector<float> cls_8;
            std::vector<float> dis_16;
            std::vector<float> cls_16;
            std::vector<float> dis_32;
            std::vector<float> cls_32;

            onnx_net->CopyFromDeviceToHost( cls_8, 4);
            onnx_net->CopyFromDeviceToHost( dis_8, 3);
            onnx_net->CopyFromDeviceToHost(cls_16, 2);            
            onnx_net->CopyFromDeviceToHost(dis_16, 1);
            onnx_net->CopyFromDeviceToHost(cls_32, 5);
            onnx_net->CopyFromDeviceToHost(dis_32, 6);

            // // post process
            std::vector<std::vector<BoxInfo>> results;
            results.resize(NUM_CLASS);
            std::vector<BoxInfo> dets;



            cv::Mat  mcls_8(1600, 80, CV_32FC1,  cls_8.data());
            cv::Mat  mdis_8(1600, 32, CV_32FC1,  dis_8.data());
            cv::Mat mcls_16( 400, 80, CV_32FC1, cls_16.data());
            cv::Mat mdis_16( 400, 32, CV_32FC1, dis_16.data());
            cv::Mat mcls_32( 100, 80, CV_32FC1, cls_32.data());
            cv::Mat mdis_32( 100, 32, CV_32FC1, dis_32.data());

            // ncnn::Mat  ncls_8( mcls_8.cols,  mcls_8.rows, 1,  (void*)mcls_8.data);
            // ncnn::Mat  ndis_8( mdis_8.cols,  mdis_8.rows, 1,  (void*)mdis_8.data);
            // ncnn::Mat ncls_16(mcls_16.cols, mcls_16.rows, 1, (void*)mcls_16.data);
            // ncnn::Mat ndis_16(mdis_16.cols, mdis_16.rows, 1, (void*)mdis_16.data);
            // ncnn::Mat ncls_32(mcls_32.cols, mcls_32.rows, 1, (void*)mcls_32.data);
            // ncnn::Mat ndis_32(mdis_32.cols, mdis_32.rows, 1, (void*)mdis_32.data);
            //  ncls_8 =  ncls_8.clone();
            //  ndis_8 =  ndis_8.clone();
            // ncls_16 = ncls_16.clone();
            // ndis_16 = ndis_16.clone();
            // ncls_32 = ncls_32.clone();
            // ndis_32 = ndis_32.clone();
            decode_infer( cls_8.data(),  dis_8.data(),  8, kCONF_THRESH, results);
            decode_infer(cls_16.data(), dis_16.data(), 16, kCONF_THRESH, results);
            decode_infer(cls_32.data(), dis_32.data(), 32, kCONF_THRESH, results);
            // std::cout << "cls:" << results.size() << "  conf:" << results[0].size() << std::endl;
            for (int i = 0; i < (int)results.size(); i++)
            {
                nms(results[i], kNMS_THRESH);
                
                for (auto box : results[i])
                {
                    dets.push_back(box);
                }
            }
            draw_bboxes(img, dets, effect_roi);
            cv::waitKey(1);



            // int kElem = INPUT_H * INPUT_W;
            // std::vector<float> rr(kElem);
            // std::vector<float> gg(kElem);
            // std::vector<float> bb(kElem);
            // for (int j = 0; j < kElem; j++)
            // {
            //     bb[j] = (float)(output[j] * 255.0);
            //     gg[j] = (float)(output[j + kElem] * 255.0);
            //     rr[j] = (float)(output[j + 2 * kElem] * 255.0);
            // }

            // cv::Mat channel[3];
            // channel[0] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, rr.data());
            // channel[1] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, gg.data());
            // channel[2] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, bb.data());
            // auto image = cv::Mat(INPUT_H, INPUT_W, CV_32FC3);
            // cv::merge(channel, 3, image);
            // cv::Mat output_image;
            // cv::resize(image, output_image, img.size(), 0, 0, cv::INTER_LINEAR);

            // auto time_end = std::chrono::steady_clock::now();
            // auto all_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
            // std::cout << "Whole time: " << all_duration << " ms. TRT enqueue time: " << duration << " ms. FPS: " << (1000 / all_duration) << std::endl;

            // cv::imwrite("../01_test.jpg", output_image);
            // cv::imshow("low light video enhanced", image_out.Img);
        }
    }
    else if (test_video)
    {
        const char* video_url = "../file/test3.avi";
        cv::VideoCapture cap(video_url);
        if (!cap.isOpened()) std::cout << "video open failed!" << std::endl;
        cv::Mat img;
        while (1)
        {
            cap >> img;
            if (img.empty()) std::cerr << "Read image failed!" << std::endl;

            auto time_start = std::chrono::steady_clock::now();

            int gpu = 1;
            int cpu = 0;
            if (gpu)
            {
                std::cout << "==========================================" << std::endl;
                cv::cuda::GpuMat input;
                cv::cuda::GpuMat output_img;
                input.upload(img);

                cv::cuda::resize(input, output_img, cv::Size(INPUT_W, INPUT_H));
            
                cv::cuda::GpuMat flt_image(INPUT_H, INPUT_W, CV_32FC3);
                output_img.convertTo(flt_image, CV_32FC3, 1.f / 255.f);

                std::vector<cv::cuda::GpuMat> chw;
                float* gpu_input = (float*)(onnx_net->mBinding[0]);
                for (size_t i = 0; i < 3; ++i)
                {
                    chw.emplace_back(cv::cuda::GpuMat(cv::Size(INPUT_W, INPUT_H), CV_32FC1, gpu_input + i * INPUT_W * INPUT_H));
                }
                cv::cuda::split(flt_image, chw);
                onnx_net->CopyFromHostToDevice(gpu_input, 0);

                // do inference
                auto time1 = std::chrono::steady_clock::now();
                onnx_net->Forward();
                auto time2 = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
                std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;
                
                // get output
                float* gpu_output = (float*)(onnx_net->mBinding[1]);
                
                cv::cuda::GpuMat flt_image_out;
                cv::cuda::GpuMat out_put;

                std::vector<cv::cuda::GpuMat> chw_1;
                for (size_t i = 0; i < 3; ++i)
                {
                    chw_1.emplace_back(cv::cuda::GpuMat(cv::Size(INPUT_W, INPUT_H), CV_32FC1, gpu_output + i * INPUT_W * INPUT_H));
                }
                cv::cuda::merge(chw_1, out_put);

                cv::cuda::GpuMat image_out;
                out_put.convertTo(image_out, CV_32FC3, 1.f * 255.f);
                cv::cuda::resize(image_out, flt_image_out, img.size());
                
                cv::Mat dst;
                flt_image_out.download(dst);

                auto time_end = std::chrono::steady_clock::now();
                auto all_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
                std::cout << "Whole time: " << all_duration << " ms. TRT enqueue time: " << duration << " ms. FPS: " << (1000 / all_duration) << std::endl;


                cv::imwrite("../files/01_1.jpg", dst);
                cv::imshow("low light video enhanced", dst);
                cv::waitKey(1);
            }
            if (cpu)
            {
                int w, h, x, y;
                float r_w = (float)(INPUT_W / (img.cols*1.0));
                float r_h = (float)(INPUT_H / (img.rows*1.0));
                if (r_h > r_w) {
                    w = INPUT_W;
                    h = (int)(r_w * img.rows);
                    x = 0;
                    y = (INPUT_H - h) / 2;
                } else {
                    w = (int)(r_h * img.cols);
                    h = INPUT_H;
                    x = (INPUT_W - w) / 2;
                    y = 0;
                }
                cv::Mat re(h, w, CV_8UC3);
                cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
                re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
                cv::resize(img, re, re.size());
                cv::Mat pr_img = out; // letterbox BGR to RGB
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row) {
                    uchar* uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col) {
                        data[i] = (float)(uc_pixel[2] / 255.0);
                        data[i + INPUT_H * INPUT_W] = (float)(uc_pixel[1] / 255.0);
                        data[i + 2 * INPUT_H * INPUT_W] = (float)(uc_pixel[0] / 255.0);
                        uc_pixel += 3;
                        ++i;
                    }
                }

                std::vector<float> input(data, data + sizeof(data)/sizeof(float));
                onnx_net->CopyFromHostToDevice(input, 0);

                // do inference
                auto time1 = std::chrono::steady_clock::now();
                onnx_net->Forward();
                auto time2 = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
                std::cout << "TRT enqueue done, time: " << duration << " ms." << std::endl;
                
                // get output
                std::vector<float> output;
                onnx_net->CopyFromDeviceToHost(output, 1);
                
                // post process
                int kElem = INPUT_H * INPUT_W;
                std::vector<float> rr(kElem);
                std::vector<float> gg(kElem);
                std::vector<float> bb(kElem);
                for (int j = 0; j < kElem; j++)
                {
                    bb[j] = (float)(output[j] * 255.0);
                    gg[j] = (float)(output[j + kElem] * 255.0);
                    rr[j] = (float)(output[j + 2 * kElem] * 255.0);
                }

                cv::Mat channel[3];
                channel[2] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, rr.data());
                channel[1] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, gg.data());
                channel[0] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, bb.data());
                auto image = cv::Mat(INPUT_H, INPUT_W, CV_32FC3);
                cv::merge(channel, 3, image);

                cv::Rect rect(x, y, w, h);
                cv::Mat image_roi = image(rect);

                auto time_end = std::chrono::steady_clock::now();
                auto all_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end-time_start).count();
                std::cout << "Whole time: " << all_duration << " ms. TRT enqueue time: " << duration << " ms. FPS: " << (1000 / all_duration) << std::endl;

                
                cv::imwrite("../01_test.jpg", image);
                // cv::imshow("low light video enhanced", image_out.Img);
            }

        }
    
    }
    else
    {
        std::cout << "choose a mode test." << std::endl;
    }
    
    return 0;
}