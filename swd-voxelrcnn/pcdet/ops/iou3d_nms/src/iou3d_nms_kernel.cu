/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <stdio.h>
#define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;
struct Point {
    float y, z;
    __device__ Point() {}
    __device__ Point(double _x, double _y){
        y = _y, z = _z;
    }

    __device__ void set(float _x, float _y){
        y = _y; z = _z;
    }

    __device__ Point operator +(const Point &b)const{
        return Point(y + b.y, z + b.z);
    }

    __device__ Point operator -(const Point &b)const{
        return Point(y - b.y, z - b.z);
    }
};

__device__ inline float cross(const Point &a, const Point &b){
    return a.y * b.z - a.z * b.y;
}

__device__ inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.y - p0.y) * (p2.z - p0.z) - (p2.y - p0.y) * (p1.z - p0.z);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.y,p2.y) <= max(q1.y,q2.y)  &&
              min(q1.y,q2.y) <= max(p1.y,p2.y) &&
              min(p1.z,p2.z) <= max(q1.z,q2.z) &&
              min(q1.z,q2.z) <= max(p1.z,p2.z);
    return ret;
}

__device__ inline int check_in_box2d(const float *box, const Point &p){
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_y = box[1], center_z = box[2];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_y = (p.y - center_y) * angle_cos + (p.z - center_z) * (-angle_sin);
    float rot_z = (p.y - center_y) * angle_sin + (p.z - center_z) * angle_cos;

    return (fabs(rot_y) < box[4] / 2 + MARGIN && fabs(rot_z) < box[5] / 2 + MARGIN);
}

__device__ inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
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

__device__ inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_y = (p.y - center.y) * angle_cos + (p.z - center.z) * (-angle_sin) + center.y;
    float new_z = (p.y - center.y) * angle_sin + (p.z - center.z) * angle_cos + center.z;
    p.set(new_y, new_z);
}

__device__ inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return -atan2(a.y - center.y, a.z - center.z) > -atan2(b.y - center.y, b.z - center.z);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]

    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2, a_dz_half = box_a[5] / 2, b_dz_half = box_b[5] / 2;
    float a_y1 = box_a[1] - a_dy_half, a_z1 = box_a[2] - a_dz_half;
    float a_y2 = box_a[1] + a_dy_half, a_z2 = box_a[2] + a_dz_half;
    float b_y1 = box_b[1] - b_dy_half, b_z1 = box_b[2] - b_dz_half;
    float b_y2 = box_b[1] + b_dy_half, b_z2 = box_b[2] + b_dz_half;

    Point center_a(box_a[1], box_a[2]);
    Point center_b(box_b[1], box_b[2]);

#ifdef DEBUG
    printf("a: (%.3f, %.3f, %.3f, %.3f, %.3f), b: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", a_y1, a_z1, a_y2, a_z2, a_angle,
           b_y1, b_z1, b_y2, b_z2, b_angle);
    printf("center a: (%.3f, %.3f), b: (%.3f, %.3f)\n", center_a.x, center_a.y, center_b.x, center_b.y);
#endif

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
#ifdef DEBUG
        printf("before corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
#ifdef DEBUG
        printf("corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
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
#ifdef DEBUG
                printf("Cross points (%.3f, %.3f): a(%.3f, %.3f)->(%.3f, %.3f), b(%.3f, %.3f)->(%.3f, %.3f) \n",
                    cross_points[cnt - 1].x, cross_points[cnt - 1].y,
                    box_a_corners[i].x, box_a_corners[i].y, box_a_corners[i + 1].x, box_a_corners[i + 1].y,
                    box_b_corners[i].x, box_b_corners[i].y, box_b_corners[i + 1].x, box_b_corners[i + 1].y);
#endif
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
#ifdef DEBUG
                printf("b corners in a: corner_b(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
#ifdef DEBUG
                printf("a corners in b: corner_a(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
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

#ifdef DEBUG
    printf("cnt=%d\n", cnt);
    for (int i = 0; i < cnt; i++){
        printf("All cross point %d: (%.3f, %.3f)\n", i, cross_points[i].x, cross_points[i].y);
    }
#endif

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

__device__ inline float iou_bev(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]
    float sa = box_a[4] * box_a[5];
    float sb = box_b[4] * box_b[5];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

__global__ void boxes_overlap_kernel(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    const int a_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    const int b_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if (a_idx >= num_a || b_idx >= num_b){
        return;
    }
    const float * cur_box_a = boxes_a + a_idx * 7;
    const float * cur_box_b = boxes_b + b_idx * 7;
    float s_overlap = box_overlap(cur_box_a, cur_box_b);
    ans_overlap[a_idx * num_b + b_idx] = s_overlap;
}

__global__ void boxes_iou_bev_kernel(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    const int a_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    const int b_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if (a_idx >= num_a || b_idx >= num_b){
        return;
    }

    const float * cur_box_a = boxes_a + a_idx * 7;
    const float * cur_box_b = boxes_b + b_idx * 7;
    float cur_iou_bev = iou_bev(cur_box_a, cur_box_b);
    ans_iou[a_idx * num_b + b_idx] = cur_iou_bev;
}

__global__ void nms_kernel(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, unsigned long long *mask){
    //params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 7 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 0];
        block_boxes[threadIdx.x * 7 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 1];
        block_boxes[threadIdx.x * 7 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 2];
        block_boxes[threadIdx.x * 7 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 3];
        block_boxes[threadIdx.x * 7 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 4];
        block_boxes[threadIdx.x * 7 + 5] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 5];
        block_boxes[threadIdx.x * 7 + 6] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 6];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 7;

        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (iou_bev(cur_box, block_boxes + i * 7) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}


__device__ inline float iou_normal(float const * const a, float const * const b) {
    //params: a: [x, y, z, dx, dy, dz, heading]
    //params: b: [x, y, z, dx, dy, dz, heading]

    float left = fmaxf(a[1] - a[4] / 2, b[1] - b[4] / 2), right = fminf(a[1] + a[4] / 2, b[1] + b[4] / 2);
    float top = fmaxf(a[2] - a[5] / 2, b[2] - b[5] / 2), bottom = fminf(a[2] + a[5] / 2, b[2] + b[5] / 2);
    float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
    float interS = width * height;
    float Sa = a[4] * a[5];
    float Sb = b[4] * b[5];
    return interS / fmaxf(Sa + Sb - interS, EPS);
}


__global__ void nms_normal_kernel(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, unsigned long long *mask){
    //params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 7 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 0];
        block_boxes[threadIdx.x * 7 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 1];
        block_boxes[threadIdx.x * 7 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 2];
        block_boxes[threadIdx.x * 7 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 3];
        block_boxes[threadIdx.x * 7 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 4];
        block_boxes[threadIdx.x * 7 + 5] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 5];
        block_boxes[threadIdx.x * 7 + 6] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 6];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 7;

        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (iou_normal(cur_box, block_boxes + i * 7) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}





void boxesoverlapLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap){

    dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK), DIVUP(num_a, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    boxes_overlap_kernel<<<blocks, threads>>>(num_a, boxes_a, num_b, boxes_b, ans_overlap);
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou){

    dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK), DIVUP(num_a, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    boxes_iou_bev_kernel<<<blocks, threads>>>(num_a, boxes_a, num_b, boxes_b, ans_iou);
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}


void nmsLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh){
    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
                DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
    dim3 threads(THREADS_PER_BLOCK_NMS);
    nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes, mask);
}


void nmsNormalLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh){
    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
                DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
    dim3 threads(THREADS_PER_BLOCK_NMS);
    nms_normal_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes, mask);
}
