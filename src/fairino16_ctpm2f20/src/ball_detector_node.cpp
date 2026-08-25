#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

// PCL 和 ROS 转换库
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// PCL 滤波和分割
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>

// TF2 转换
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/exceptions.h>
#include <chrono>

class BallDetector : public rclcpp::Node
{
public:
  BallDetector() : Node("ball_detector_node")
  {
    RCLCPP_INFO(this->get_logger(), "Ball Detector Node has been started.");

    // ----------------------------------------------------------------------
    // 1. 参数声明与读取
    // ----------------------------------------------------------------------
    this->declare_parameter<std::string>("target_frame", "world");
    this->declare_parameter<std::string>("input_topic", "/depth_camera/points");
    this->declare_parameter<std::string>("output_topic", "/ball_detector/ball_pose");
    
    // ROI 滤波参数
    this->declare_parameter<std::vector<double>>("roi_filter.x_range", {0.2, 0.8});
    this->declare_parameter<std::vector<double>>("roi_filter.y_range", {-0.3, 0.3});
    this->declare_parameter<std::vector<double>>("roi_filter.z_range", {0.01, 1.0});

    // RANSAC 参数 (移除桌子)
    this->declare_parameter<int>("ransac.max_iterations", 100);
    this->declare_parameter<double>("ransac.distance_threshold", 0.01);

    // 欧几里德聚类参数
    this->declare_parameter<double>("clustering.cluster_tolerance", 0.02);
    this->declare_parameter<int>("clustering.min_cluster_size", 50);
    this->declare_parameter<int>("clustering.max_cluster_size", 1000);

    // 读取参数到成员变量
    this->get_parameter("target_frame", target_frame_);
    this->get_parameter("roi_filter.x_range", x_range_);
    this->get_parameter("roi_filter.y_range", y_range_);
    this->get_parameter("roi_filter.z_range", z_range_);

    this->get_parameter("ransac.max_iterations", ransac_max_iterations_);
    this->get_parameter("ransac.distance_threshold", ransac_distance_threshold_);

    this->get_parameter("clustering.cluster_tolerance", cluster_tolerance_);
    this->get_parameter("clustering.min_cluster_size", min_cluster_size_);
    this->get_parameter("clustering.max_cluster_size", max_cluster_size_);
    
    RCLCPP_INFO(this->get_logger(), "Target frame for publishing: %s", target_frame_.c_str());

    // ----------------------------------------------------------------------
    // 2. TF2 初始化
    // ----------------------------------------------------------------------
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ----------------------------------------------------------------------
    // 3. 订阅与发布初始化
    // ----------------------------------------------------------------------
    std::string input_topic, output_topic;
    this->get_parameter("input_topic", input_topic);
    this->get_parameter("output_topic", output_topic);

    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic, 1, // 降低 QoS 队列深度，快速处理最新数据
        std::bind(&BallDetector::cloud_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
        output_topic, 1);
  }

private:
  using PointCloudT = pcl::PointCloud<pcl::PointXYZ>;

  // ----------------------------------------------------------------------
  // 成员变量
  // ----------------------------------------------------------------------
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr publisher_;

  // 参数成员变量
  std::string target_frame_;
  std::vector<double> x_range_, y_range_, z_range_;
  double cluster_tolerance_;
  int min_cluster_size_, max_cluster_size_;
  int ransac_max_iterations_;
  double ransac_distance_threshold_;


  // ----------------------------------------------------------------------
  // 核心回调函数
  // ----------------------------------------------------------------------
  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // A. 转换 ROS PointCloud2 到 PCL 结构
    PointCloudT::Ptr cloud_in (new PointCloudT);
    pcl::fromROSMsg(*msg, *cloud_in);

    if (cloud_in->empty()) {
        RCLCPP_WARN(this->get_logger(), "Received empty point cloud.");
        return;
    }
    
    PointCloudT::Ptr cloud_filtered (new PointCloudT);
    *cloud_filtered = *cloud_in; 


    // B. PassThrough ROI 滤波 (限制感兴趣区域)
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud_filtered); 
    
    // X 轴过滤 (前后)
    pass.setFilterFieldName("x");
    pass.setFilterLimits(x_range_[0], x_range_[1]);
    pass.filter(*cloud_filtered);
    
    // Y 轴过滤 (左右)
    pass.setFilterFieldName("y");
    pass.setFilterLimits(y_range_[0], y_range_[1]);
    pass.filter(*cloud_filtered);
    
    // Z 轴过滤 (上下)
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_range_[0], z_range_[1]);
    pass.filter(*cloud_filtered);

    if (cloud_filtered->empty()) {
        RCLCPP_DEBUG(this->get_logger(), "ROI filter resulted in an empty cloud.");
        return;
    }

    // C. RANSAC 平面分割 (识别并移除桌子)
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE); 
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(ransac_max_iterations_);
    seg.setDistanceThreshold(ransac_distance_threshold_); 

    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        RCLCPP_WARN(this->get_logger(), "Could not estimate a planar model (table).");
        return;
    }

    // 提取非平面点 (即桌子上的物体)
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(true); // 提取不在平面上的点
    
    PointCloudT::Ptr cloud_objects (new PointCloudT);
    extract.filter(*cloud_objects);

    if (cloud_objects->empty()) {
        RCLCPP_DEBUG(this->get_logger(), "No objects found on the table after RANSAC.");
        return;
    }

    // D. 欧几里德聚类 (找到小球)
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_objects);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_); 
    ec.setMaxClusterSize(max_cluster_size_); 
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_objects);
    ec.extract(cluster_indices);

    // E. 计算质心并发布
    if (cluster_indices.empty()) {
        RCLCPP_INFO(this->get_logger(), "No ball clusters found matching criteria.");
        return;
    }

    // 假设第一个找到的簇就是我们想要的小球
    const auto& indices = cluster_indices[0]; 
    PointCloudT::Ptr ball_cluster (new PointCloudT);
    
    // 从对象点云中提取第一个簇
    for (const int& index : indices.indices) {
        ball_cluster->push_back(cloud_objects->points[index]);
    }
    
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*ball_cluster, centroid);

    RCLCPP_INFO(this->get_logger(), "Ball detected at (Camera Frame): x=%.3f, y=%.3f, z=%.3f",
                centroid[0], centroid[1], centroid[2]);
    
    // 发布坐标 (包含 TF 转换)
    publish_ball_pose(msg->header.frame_id, centroid);
  }


  // ----------------------------------------------------------------------
  // 辅助函数：坐标系转换和发布
  // ----------------------------------------------------------------------
  void publish_ball_pose(const std::string& source_frame, const Eigen::Vector4f& centroid)
  {
    geometry_msgs::msg::PointStamped source_point;
    source_point.header.frame_id = source_frame;
    source_point.header.stamp = this->now();
    source_point.point.x = centroid[0];
    source_point.point.y = centroid[1];
    source_point.point.z = centroid[2];

    geometry_msgs::msg::PointStamped transformed_point;

    // 使用 TF2 进行坐标系转换
    try 
    {
        // 等待目标坐标系可用，并进行转换
        if (tf_buffer_->canTransform(target_frame_, source_frame, tf2::TimePointZero, std::chrono::milliseconds(100)))
        {
             tf_buffer_->transform(source_point, transformed_point, target_frame_); 
        } else {
             RCLCPP_WARN(this->get_logger(), "Waiting for transform from %s to %s...",
                        source_frame.c_str(), target_frame_.c_str());
             return;
        }

        RCLCPP_INFO(this->get_logger(), "Ball transformed to (%s): x=%.3f, y=%.3f, z=%.3f",
                    target_frame_.c_str(), transformed_point.point.x, transformed_point.point.y, transformed_point.point.z);
        
        // 发布最终坐标
        publisher_->publish(transformed_point);
    } 
    catch (const tf2::TransformException & ex) 
    {
        RCLCPP_WARN(this->get_logger(), "Could not transform point from %s to %s: %s",
                    source_frame.c_str(), target_frame_.c_str(), ex.what());
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BallDetector>());
  rclcpp::shutdown();
  return 0;
}