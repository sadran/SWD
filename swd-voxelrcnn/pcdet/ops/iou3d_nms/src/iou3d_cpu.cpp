/*
3D Rotated IoU Calculation (CPU)
Written by Shaoshuai Shi
All Rights Reserved 2020.
*/

#include <stdio.h>
#include <math.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "iou3d_cpu.h"

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

inline float min(float a, float b){
    return a > b ? b : a;
}

inline float max(float a, float b){
    return a > b ? a : b;
}

const float EPS = 1e-8;
struct Point {
    float y, z;
    __device__ Point() {}
    __device__ Point(double _y, double _z){
        y = _y, z = _z;
    }

    __device__ void set(float _y, float _z){
        y = _y; z = _z;
    }

    __device__ Point operator +(const Point &b)const{
        return Point(y + b.y, z + b.z);
    }

    __device__ Point operator -(const Point &b)const{
        return Point(y - b.y, z - b.z);
    }
};

inline float cross(const Point &a, const Point &b){
    return a.y * b.z - a.z * b.y;
}

inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.y - p0.y) * (p2.z - p0.z) - (p2.y - p0.y) * (p1.z - p0.z);
}

inline int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y) &&
              min(p1.z,p2.z) <= max(q1.z,q2.z) &&
              min(q1.z,q2.z) <= max(p1.z,p2.z);
    return ret;
}

inline int check_in_box2d(const float *box, const Point &p){
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_y = box[1], center_z = box[2];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_y = (p.y - center_y) * angle_cos + (p.z - center_z) * (-angle_sin);
    float rot_z = (p.y - center_y) * angle_sin + (p.z - center_z) * angle_cos;

    return (fabs(rot_y) < box[4] / 2 + MARGIN && fabs(rot_z) < box[5] / 2 + MARGIN);
}

inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
        ans.z = (s5 * q0.z - s1 * q1.z) / (s5 - s1);
    }
    else{
        float a0 = p0.z - p1.z, b0 = p1.y - p0.y, c0 = p0.y * p1.z - p1.y * p0.z;
        float a1 = q0.z - q1.z, b1 = q1.y - q0.y, c1 = q0.y * q1.z - q1.y * q0.z;
        float D = a0 * b1 - a1 * b0;

        ans.y = (b0 * c1 - b1 * c0) / D;
        ans.z = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_y = (p.y - center.y) * angle_cos + (p.z - center.z) * (-angle_sin) + center.y;
    float new_z = (p.y - center.y) * angle_sin + (p.z - center.z) * angle_cos + center.z;
    p.set(new_y, new_z);
}

inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return -atan2(a.y - center.y, a.z - center.z) > -atan2(b.y - center.y, b.z - center.z);
}

inline float box_overlap(const float *box_a, const float *box_b){
    // params: box_a (7) [x, y, z, dx, dy, dz, heading]
    // params: box_b (7) [x, y, z, dx, dy, dz, heading]

    //float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3], a_angle = box_a[4];
    //float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3], b_angle = box_b[4];
    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2, a_dz_half = box_a[5] / 2, b_dz_half = box_b[5] / 2;
    float a_y1 = box_a[1] - a_dy_half, a_z1 = box_a[2] - a_dz_half;
    float a_y2 = box_a[1] + a_dy_half, a_z2 = box_a[2] + a_dz_half;
    float b_y1 = box_b[1] - b_dy_half, b_z1 = box_b[2] - b_dz_half;
    float b_y2 = box_b[1] + b_dy_half, b_z2 = box_b[2] + b_dz_half;

    Point center_a(box_a[1], box_a[2]);
    Point center_b(box_b[1], box_b[2]);

    Point box_a_corners[5];
    box_a_corners[0].set(a_y1, a_z1);
    box_a_corners[1].set(a_y2, a_z1);
    box_a_corners[2].set(a_y2, a_z2);
    box_a_corners[3].set(a_y1, a_z2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_y1, b_z1);
    box_b_corners[1].set(b_y2, b_z1);
    box_b_corners[2].set(b_y2, b_z2);
    box_b_corners[3].set(b_y1, b_z2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.y /= cnt;
    poly_center.z /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

inline float iou_bev(const float *box_a, const float *box_b){
    // params: box_a (7) [x, y, z, dx, dy, dz, heading]
    // params: box_b (7) [x, y, z, dx, dy, dz, heading]
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}


int boxes_iou_bev_cpu(at::Tensor boxes_a_tensor, at::Tensor boxes_b_tensor, at::Tensor ans_iou_tensor){
    // params boxes_a_tensor: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b_tensor: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_iou_tensor: (N, M)

    CHECK_CONTIGUOUS(boxes_a_tensor);
    CHECK_CONTIGUOUS(boxes_b_tensor);

    int num_boxes_a = boxes_a_tensor.size(0);
    int num_boxes_b = boxes_b_tensor.size(0);
    const float *boxes_a = boxes_a_tensor.data<float>();
    const float *boxes_b = boxes_b_tensor.data<float>();
    float *ans_iou = ans_iou_tensor.data<float>();

    for (int i = 0; i < num_boxes_a; i++){
        for (int j = 0; j < num_boxes_b; j++){
            ans_iou[i * num_boxes_b + j] = iou_bev(boxes_a + i * 7, boxes_b + j * 7);
        }
    }
    return 1;
}
