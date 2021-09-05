#include <string>
#include <sstream>
#include <cmath>
#include <Poco/Logger.h>
#include <Poco/Util/Application.h>
#include <Poco/Util/AbstractConfiguration.h>
#include <Eigen/Core>
#include <nanogui/common.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <nanogui/button.h>
#include <nanogui/messagedialog.h>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "MainWindow.h"
#include "VideoWindow.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mqtt/async_client.h"
#include <exception>

const std::string DFLT_SERVER_ADDRESS{ "ws://192.168.1.2:1884" };
const int QOS = 1;
const auto TIMEOUT = std::chrono::seconds(10);
const std::string address = DFLT_SERVER_ADDRESS;
bool is_true = FALSE;
using std::string;
using std::mutex;
using std::lock_guard;
using std::ostringstream;
using Poco::Logger;
using Poco::Util::Application;
using Poco::Util::AbstractConfiguration;
using Eigen::Vector2i;
using nanogui::Screen;
using nanogui::Window;
using nanogui::Color;
using nanogui::GroupLayout;
using nanogui::Label;
using nanogui::Button;
using nanogui::MessageDialog;

MainWindow::MainWindow(const Vector2i & size, const string & caption)
    : Screen(size, caption)
    , _logger{ Logger::get("MainWindow") }
    , _config(Application::instance().config())
    , _isVideoStarted{ false }
    , _colorRatio{ 4.0f / 3.0f }
    , _depthRatio{ 4.0f / 3.0f }
    , _align(RS2_STREAM_COLOR)
    , _inWidth{ 300 }
    , _inHeight{ 300 }
    , _inScaleFactor{ 0.007843f }
    , _meanVal{ 127.5f }
    , _classNames{ "background",  "robot" }


{
    
    // initialize text translation table
    initTextMap();
    // setting & configuration window
    _settingWindow = new Window(this, _textmap[TextId::ControlSetting]);
    _settingWindow->setPosition(Vector2i(0, 0));
    _settingWindow->setLayout(new GroupLayout());

    new Label(_settingWindow, _textmap[TextId::VideoStream], "sans-bold");
    _btnColorStream = _settingWindow->add<Button>(_textmap[TextId::ColorStream]);
    _btnColorStream->setBackgroundColor(Color(0, 0, 255, 25));
    _btnColorStream->setFlags(Button::ToggleButton);
    _btnColorStream->setTooltip("Show RGB color video stream");
    _btnColorStream->setChangeCallback([&](bool state) { onToggleColorStream(state); });

    _btnDepthStream = _settingWindow->add<Button>(_textmap[TextId::DepthStream]);
    _btnDepthStream->setBackgroundColor(Color(0, 0, 255, 25));
    _btnDepthStream->setFlags(Button::ToggleButton);
    _btnDepthStream->setTooltip("Show depth sensor stream");
    _btnDepthStream->setChangeCallback([&](bool state) { onToggleDepthStream(state); });

    new Label(_settingWindow, _textmap[TextId::DnnObjDetect], "sans-bold");
    _btnStartCvdnn = _settingWindow->add<Button>(_textmap[TextId::StartDetect]);
    _btnStartCvdnn->setBackgroundColor(Color(255, 0, 128, 25));
    _btnStartCvdnn->setFlags(Button::ToggleButton);
    _btnStartCvdnn->setTooltip("Start MobileNet Single-Shot Detector");
    _btnStartCvdnn->setChangeCallback([&](bool state) { onToggleCvdnn(state); });

    _colorWindow = nullptr;
    _depthWindow = nullptr;

    performLayout();

    // load trained DNN model
    _net = cv::dnn::readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel");


  
    


}

void MainWindow::onToggleColorStream(bool on)
{
    if (on && !tryStartVideo())
    {
        _btnColorStream->setPushed(false);
        return;
    }

    if (on && _colorWindow == nullptr)
    {
        _colorWindow = new VideoWindow(this, _textmap[TextId::ColorStream]);
        _colorWindow->setPosition(Vector2i(_settingWindow->size()(0), 0));
        performLayout();
        resizeEvent(this->size());
    }
    else if (!on && _colorWindow != nullptr)
    {
        _colorWindow->dispose();
        _colorWindow = nullptr;
        if (_depthWindow == nullptr)
            stopVideo();
    }
}

void MainWindow::onToggleDepthStream(bool on)
{
    if (on && !tryStartVideo())
    {
        _btnDepthStream->setPushed(false);
        return;
    }

    if (on && _depthWindow == nullptr)
    {
        _depthWindow = new VideoWindow(this, _textmap[TextId::DepthStream]);
        _depthWindow->setPosition(Vector2i(_settingWindow->size()(0) + 30, 30));
        performLayout();
        resizeEvent(this->size());
    }
    else if (!on && _depthWindow != nullptr)
    {
        _depthWindow->dispose();
        _depthWindow = nullptr;
        if (_colorWindow == nullptr)
            stopVideo();
    }
}

void MainWindow::onToggleCvdnn(bool on)
{
    if (on && !isVideoStarted())
    {
        new MessageDialog(this, MessageDialog::Type::Warning, "Warning", "Please start playing color or depth stream before DNN detector.");
        _btnStartCvdnn->setPushed(false);
        return;
    }

    lock_guard<mutex> guard{ _mutex };
    _isCvdnnStarted = on;
}

bool MainWindow::keyboardEvent(int key, int scancode, int action, int modifiers)
{
    if (Screen::keyboardEvent(key, scancode, action, modifiers))
        return true;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        setVisible(false);
        return true;
    }
    return false;
}

bool MainWindow::resizeEvent(const Eigen::Vector2i & size)
{
    // rescale setting window, the minimum width 150 is expected to be good enough, only height is resized
    Vector2i new_setting_size = Vector2i(std::max(150, _settingWindow->size()(0)), size(1) - 3);
    _settingWindow->setSize(new_setting_size);

    // color window take full available space
    if (_colorWindow != nullptr)
        _colorWindow->setSize(Vector2i(size(0) - new_setting_size(0) - 3, size(1) - 15));

    if (_depthWindow != nullptr)
    {
        float depth2ScreenRatio = 0.25f;
        // rescale depth window to fit with depth frame
        Vector2i new_depth_size(0, 0);
        if (_depthRatio > ((float)size(0) / size(1)))
        {
            // space width is smaller than expected ratio
            new_depth_size(0) = std::lround(size(0) * depth2ScreenRatio);
            new_depth_size(1) = std::lround(new_depth_size(0) / _depthRatio);
        }
        else
        {
            // space height is smaller than expected ratio
            new_depth_size(1) = std::lround(size(1) * depth2ScreenRatio);
            new_depth_size(0) = std::lround(new_depth_size(1) * _depthRatio);
        }
        _depthWindow->setSize(new_depth_size);
    }

    return true;
}

void MainWindow::draw(NVGcontext * ctx)
{
    if (isVideoStarted())
    {
        rs2::frameset frames = _pipe.wait_for_frames();
        frames = _align.process(frames);
        rs2::video_frame colorFrame = frames.get_color_frame();
        rs2::depth_frame depthFrame = frames.get_depth_frame();

        if (isCvdnnStarted())
            detectObjects(colorFrame, depthFrame, _depthScale);

        if (_colorWindow != nullptr)
            _colorWindow->setVideoFrame(colorFrame);

       
    }

    Screen::draw(ctx);
}

void MainWindow::initTextMap()
{
    // initialize the text translation table
    string lang = _config.getString("application.language", "en_US");
    _textmap[TextId::ControlSetting] = _config.getString(lang + ".ControlSetting", "Control / Setting");
    _textmap[TextId::VideoStream] = _config.getString(lang + ".VideoStream", "Video Stream");
    _textmap[TextId::ColorStream] = _config.getString(lang + ".ColorStream", "Color Stream");
    _textmap[TextId::DepthStream] = _config.getString(lang + ".DepthStream", "Depth Stream");
    _textmap[TextId::DnnObjDetect] = _config.getString(lang + ".DnnObjDetect", "DNN Object Detection");
    _textmap[TextId::StartDetect] = _config.getString(lang + ".StartDetect", "Start Detecting");
}

bool MainWindow::tryStartVideo()
{
    lock_guard<mutex> guard{ _mutex };

    if (_isVideoStarted)
        return true;

    try
    {
        // check if realsense device connected
        rs2::context ctx;
        if (ctx.query_devices().size() == 0)
        {
            new MessageDialog(this, MessageDialog::Type::Warning, "Warning", "No RealSense device is found, please connect the device and try again.");
            return false;
        }

        // Even though both streams are manually configured here,
        // the depth frame will have the same size as color frame after align process
        rs2::config config;
        config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
        config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        // Start streaming with configured streams
        auto profile = _pipe.start(config).get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
     
        
        _depthScale = _pipe.get_active_profile().get_device().first<rs2::depth_sensor>().get_depth_scale();

        // calculate the proper crop size and region for DNN model to work
        float whRatio = (float)_inWidth / _inHeight;
        cv::Size cropSize = ((float)profile.width() / profile.height()) > whRatio ?
            cv::Size(static_cast<int>(profile.height() * whRatio), profile.height()) :
            cv::Size(profile.width(), static_cast<int>(profile.width() / whRatio));
        cv::Point ptRoiLt((profile.width() - cropSize.width) / 2, (profile.height() - cropSize.height) / 2);
        _rectRoi = cv::Rect(ptRoiLt, cropSize);
        _rectRoiLeft = cv::Rect(0, 0, _rectRoi.tl().x - 1, cropSize.height);
        _rectRoiRight = cv::Rect(_rectRoi.br().x + 1, 0, profile.width() - _rectRoi.br().x - 1, cropSize.height);
    

        _isVideoStarted = true;
        return true;
    }
    catch (const rs2::error & e)
    {
        ostringstream errmsg;
        errmsg << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what();
        poco_error(_logger, errmsg.str());
        return false;
    }
    catch (const std::exception & e)
    {
        poco_error(_logger, string(e.what()));
        return false;
    }
}

void MainWindow::stopVideo()
{
    lock_guard<mutex> guard{ _mutex };

    if (!_isVideoStarted)
        return;

    try
    {
        _isVideoStarted = false;
        _pipe.stop();
    }
    catch (const rs2::error & e)
    {
        ostringstream errmsg;
        errmsg << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what();
        poco_error(_logger, errmsg.str());
        return;
    }
    catch (const std::exception & e)
    {
        poco_error(_logger, string(e.what()));
        return;
    }
}

bool MainWindow::isVideoStarted()
{
    lock_guard<mutex> guard{ _mutex };
    return _isVideoStarted;
}

bool MainWindow::isCvdnnStarted()
{
    lock_guard<mutex> guard{ _mutex };
    return _isCvdnnStarted;
}
float MainWindow::dist_3d(rs2::depth_frame depth_frame,int x1,int y1,int x2,int y2)
{
    float upixel[2]; // From pixel
    float upoint[3]; // From point (in 3D)

    float vpixel[2]; // To pixel
    float vpoint[3]; // To point (in 3D)
    upixel[0] = x1;
    upixel[1] = y1;
    vpixel[0] = x2;
    vpixel[1] = y2;
    auto udist = depth_frame.get_distance(upixel[0], upixel[1]);
    auto vdist = depth_frame.get_distance(vpixel[0], vpixel[1]);
    rs2_intrinsics intr = depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics(); // Calibration data
    rs2_deproject_pixel_to_point(upoint, &intr, upixel, udist);
    rs2_deproject_pixel_to_point(vpoint, &intr, vpixel, vdist);
    return sqrt(pow(upoint[0] - vpoint[0], 2) +pow(upoint[1] - vpoint[1], 2) +pow(upoint[2] - vpoint[2], 2));
}
void MainWindow::detectObjects(rs2::video_frame color_frame, rs2::depth_frame depth_frame, float depth_scale)
{
    // convert RealSense frame to OpenCV Mat
    cv::Mat matColor(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::cvtColor(matColor, matColor, cv::COLOR_RGB2BGR);
    // convert and clone depth frame to OpenCV Mat
    cv::Mat matDepth = cv::Mat(cv::Size(depth_frame.get_width(), depth_frame.get_height()), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP).clone();
    matDepth.convertTo(matDepth, CV_64F);
    matDepth = matDepth * _depthScale;

    // convert mat to batch of images
    cv::Mat inputBlob = cv::dnn::blobFromImage(matColor, _inScaleFactor, cv::Size((int)_inWidth, (int)_inHeight), _meanVal, false);
    // set the network input
    _net.setInput(inputBlob, "data");
    // compute output
    cv::Mat detection = _net.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    // crop the input frame
    cv::Mat matColorRoi = matColor(_rectRoi);
    matDepth = matDepth(_rectRoi);

  

   
    auto connBuilder = mqtt::connect_options_builder();


    mqtt::async_client client(address, "new");
    auto connOpts = connBuilder
        .keep_alive_interval(std::chrono::seconds(45))
        .automatic_reconnect()
        .finalize();
    //if (is_true == FALSE)
    //{ 
        client.connect(connOpts)->wait();
      //  is_true = TRUE;
    //}


    
    //float x_size = dist_3d(depth_frame,0, 0, 0, 479);
    //float y_size = dist_3d(depth_frame, 0, 0, 839, 0);
    //float confidenceThreshold = 0.1f;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
         

            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * matColorRoi.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * matColorRoi.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * matColorRoi.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * matColorRoi.rows);

            cv::Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));
            object = object & cv::Rect(0, 0, matDepth.cols, matDepth.rows);

            // mean depth inside the detection region
            int nzCount = cv::countNonZero(matDepth(object));
            double meanDistance = (nzCount > 0) ? cv::sum(matDepth(object))[0] / nzCount : 0.0;
            std::ostringstream ssout;
            int approx_x_pixel = (xLeftBottom + xRightTop) / 2;
            int aprrox_y_pixel = (yLeftBottom + yRightTop) / 2;
            
            ssout << "<" << _classNames[objectClass]<< (xLeftBottom+xRightTop)/2<<","<<(480-(yRightTop+yLeftBottom)/2) << "> : ";
            //ssout << "<" << x_size << "," << y_size << ">";
            if (_classNames[objectClass] == "robot")
            {
                int x = (xLeftBottom + xRightTop) / 2;
                int y = (480 - (yRightTop + yLeftBottom) / 2);
                auto message = std::to_string(x);
                auto message2 = std::to_string(y);
                auto msg = mqtt::make_message("tag/x", message, 0,TRUE);
                auto msg2 = mqtt::make_message("tag/y", message2, 0, TRUE);
             
                    client.publish(msg);
                    client.publish(msg2);
                
            }
            cv::rectangle(matColorRoi, object, cv::Scalar(0, 255, 0));
            int baseLine = 0;
            cv::Size labelSize = getTextSize(ssout.str(), cv::FONT_HERSHEY_COMPLEX, 0.6, 2, &baseLine);
            cv::Point ptCenter = (object.br() + object.tl()) * 0.5;
            ptCenter.x = ptCenter.x - labelSize.width / 2;
            cv::rectangle(matColorRoi,
                cv::Rect(cv::Point(ptCenter.x, ptCenter.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)),
                cv::Scalar(128, 255, 128), CV_FILLED);
            putText(matColorRoi, ssout.str(), ptCenter, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }
    }

    cv::cvtColor(matColorRoi, matColorRoi, cv::COLOR_BGR2RGB);
    // gray out the left of ROI
    //cv::Mat matColorRoiLeft = matColor(_rectRoiLeft);
    //cv::Mat matGrayLeft;
    //cv::cvtColor(matColorRoiLeft, matGrayLeft, cv::COLOR_RGB2GRAY);
    //cv::cvtColor(matGrayLeft, matColorRoiLeft, cv::COLOR_GRAY2RGB);
    // gray out the roght of ROI
    //cv::Mat matColorRoiRight = matColor(_rectRoiRight);
    //cv::Mat matGrayRight;
    //cv::cvtColor(matColorRoiRight, matGrayRight, cv::COLOR_RGB2GRAY);
    //cv::cvtColor(matGrayRight, matColorRoiRight, cv::COLOR_GRAY2RGB);
}
