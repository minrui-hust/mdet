#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <utility>

namespace geometry2d{


__device__ const float EPS = 1e-8;

struct __attribute__((packed)) Vector{
  float x, y;

  __device__ Vector() {}
  __device__ Vector(float _x, float _y):x(_x),y(_y) {}

  __device__ Vector operator+(const Vector &b) const {
    return Vector(x + b.x, y + b.y);
  }

  __device__ Vector operator-(const Vector &b) const {
    return Vector(x - b.x, y - b.y);
  }

  __device__ float cross(const Vector &other) const{
    return x * other.y - y * other.x;
  }
};
static_assert(sizeof(Vector)==8, "Vector size is not 8");

using Point = Vector;

struct __attribute__((packed)) Rotation{
  float real, img; // 2d rotation reperesent by complex number

  __device__ Rotation() {}
  __device__ Rotation(float _real, float _img):real(_real),img(_img) {}

  __device__ Rotation inverse() const{
    return Rotation(real, -img);
  }

  __device__ Vector act(const Vector& v) const{
    float x = real * v.x - img  * v.y;
    float y = img  * v.x + real * v.y;
    return Vector(x, y);
  }
};
static_assert(sizeof(Rotation)==8, "Vector size is not 8");

namespace {

__device__ inline int point_cmp(const Point &a, const Point &b,
                                const Point &center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}

template<typename T>
__device__ void swap(T& a, T& b){
  T c;
  c = a;
  a = b;
  b = c;
}

}

struct __attribute__((packed)) Line{
  Point start, end;

  __device__ Line(){}
  __device__ Line(const Point&_start, const Point&_end):start(_start), end(_end){}

  __device__ std::pair<bool, Point> intersect(const Line& other) const{
    std::pair<bool, Point> ret(false, Point());

    const auto& p0 = this->start;
    const auto& p1 = this->end;
    const auto& q0 = other.start;
    const auto& q1 = other.end;

    // fast exclusion
    if (min(p0.x, p1.x) > max(q0.x, q1.x) ||
        min(q0.x, q1.x) > max(p0.x, p1.x) ||
        min(p0.y, p1.y) > max(q0.y, q1.y) ||
        min(q0.y, q1.y) > max(p0.y, p1.y) ){
      return ret;
    }

    float s1 = (q0 - p0).cross(p1 - p0);
    float s2 = (p1 - p0).cross(q1 - p0);
    float s3 = (p0 - q0).cross(q1 - q0);
    float s4 = (q1 - q0).cross(p1 - q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)){
      return ret;
    }

    float s5 = (q1 - p0).cross(p1 - p0);
    if (abs(s5 - s1) > EPS) {
      ret.second.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
      ret.second.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    } else {
      float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
      float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
      float D = a0 * b1 - a1 * b0;
      ret.second.x = (b0 * c1 - b1 * c0) / D;
      ret.second.y = (a1 * c0 - a0 * c1) / D;
    }
    ret.first = true;

    return ret;

  }
};
static_assert(sizeof(Line)==16, "Line size is not 16");

struct __attribute__((packed)) Box{
  Vector center; // center position in global frame
  Vector extend; // half length, half width
  Rotation rotation; // cos, sin

  __device__ Box(){}
  __device__ Box(const Vector& _center, const Vector& _extend, const Rotation& _rotation)
    :center(_center), extend(_extend), rotation(_rotation){}

  __device__ float overlap(const Box& other) const{
    const auto& box_a = *this;
    const auto& box_b = other;

    Point box_a_corners[5];
    box_a_corners[0] = box_a.rotation.act(Vector(-box_a.extend.x, -box_a.extend.y)) + box_a.center;
    box_a_corners[1] = box_a.rotation.act(Vector( box_a.extend.x, -box_a.extend.y)) + box_a.center;
    box_a_corners[2] = box_a.rotation.act(Vector( box_a.extend.x,  box_a.extend.y)) + box_a.center;
    box_a_corners[3] = box_a.rotation.act(Vector(-box_a.extend.x,  box_a.extend.y)) + box_a.center;
    box_a_corners[4] = box_a_corners[0];

    Point box_b_corners[5];
    box_b_corners[0] = box_b.rotation.act(Vector(-box_b.extend.x, -box_b.extend.y)) + box_b.center;
    box_b_corners[1] = box_b.rotation.act(Vector( box_b.extend.x, -box_b.extend.y)) + box_b.center;
    box_b_corners[2] = box_b.rotation.act(Vector( box_b.extend.x,  box_b.extend.y)) + box_b.center;
    box_b_corners[3] = box_b.rotation.act(Vector(-box_b.extend.x,  box_b.extend.y)) + box_b.center;
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center(0, 0);
    int cnt = 0;

    for (int i = 0; i < 4; i++) {
      Line line_a(box_a_corners[i], box_a_corners[i+1]);
      for (int j = 0; j < 4; j++) {
        Line line_b(box_b_corners[j], box_b_corners[j+1]);
        auto ret = line_a.intersect(line_b);
        if (ret.first) {
          cross_points[cnt] = ret.second;
          poly_center = poly_center + ret.second;
          cnt++;
        }
      }
    }

    // check corners
    for (int k = 0; k < 4; k++) {
      if (box_a.contain(box_b_corners[k])) {
        poly_center = poly_center + box_b_corners[k];
        cross_points[cnt] = box_b_corners[k];
        cnt++;
      }
      if (box_b.contain(box_a_corners[k])) {
        poly_center = poly_center + box_a_corners[k];
        cross_points[cnt] = box_a_corners[k];
        cnt++;
      }
    }

    if (cnt == 0){
      return 0;
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    for (int i = 0; i < cnt - 1; i++) {
      for (int j = 0; j < cnt - i - 1; j++) {
        if (point_cmp(cross_points[j], cross_points[j + 1], poly_center)) {
          swap(cross_points[j], cross_points[j+1]);
        }
      }
    }

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
      area += (cross_points[k] - cross_points[0]).cross(cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
  }

  __device__ bool contain(const Point& p) const{
    const float MARGIN = 1e-5;

    Point local_p = rotation.inverse().act(p-center);

    return (local_p.x > -extend.x - MARGIN && 
        local_p.x <  extend.x + MARGIN &&
        local_p.y > -extend.y - MARGIN && 
        local_p.y <  extend.y + MARGIN );
  }

  __device__ float area() const {
    return 2 *extend.x * extend.y;
  }

  static constexpr int Dim = 6;
};
static_assert(sizeof(Box)==24, "Box size is not 24");

}
