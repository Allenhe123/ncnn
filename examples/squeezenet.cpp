// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;

#if NCNN_VULKAN
    squeezenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    squeezenet.load_param("squeezenet_v1.1.param");
    squeezenet.load_model("squeezenet_v1.1.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

void load_labels(std::string& lablelfile, std::vector<std::string>& ivec)
{
    std::ifstream ifs;
    ifs.open(lablelfile.c_str());
    if (ifs.is_open())
    {
        std::string temp;
        std::string line;
        while (std::getline(ifs, line))
        {
            std::string::size_type pos = line.find_first_of(" ");
            if (pos != std::string::npos)
            {
                std::string rhs = line.substr(pos + 1, line.size() - pos - 1);
                ivec.push_back(rhs);
            }
            else
            {
                std::cout << "load label ERROR!" << std::endl;
            }
        }
        ifs.close();
    }
}


static std::string print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    std::string labelfile("synset_words.txt");
    std::vector<std::string> labels;
    load_labels(labelfile, labels);

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        std::string labelname = labels[index];
        fprintf(stderr, "%d = %f, name: %s\n", index, score, labelname.c_str());
    }

    return labels[vec[0].second];
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    std::string labelname = print_topk(cls_scores, 3);

    //获取文本框的长宽
    int baseline = 0;
	cv::Size text_size = cv::getTextSize(labelname, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
	//文本框位置
	cv::Point origin; 
	origin.x = 10;
    origin.y = text_size.height  + 10;
    cv::putText(m, labelname, origin, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, 8, 0);

    cv::imshow("img", m);

    int code = cv::waitKey(0);
    if (code == 'q')
    {
        return 0;
    }

    return 0;
}
