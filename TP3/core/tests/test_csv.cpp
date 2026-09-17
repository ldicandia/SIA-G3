#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "data/csv.hpp"
#include "doctest.h"

using tp3::LabeledImageCsv;
using tp3::PlainCsv;
using tp3::load_labeled_image_csv;
using tp3::load_plain_csv;
using tp3::parse_csv_line;

#ifndef TP3_SOURCE_DIR
#define TP3_SOURCE_DIR "."
#endif

TEST_CASE("parse_csv_line: empty, plain, and quoted with embedded commas") {
    CHECK(parse_csv_line("").empty());

    std::vector<std::string> plain = {"a", "b", "c"};
    CHECK(parse_csv_line("a,b,c") == plain);

    std::vector<std::string> quoted = {"0", "[1.0, 2.0]"};
    CHECK(parse_csv_line("0,\"[1.0, 2.0]\"") == quoted);
}

TEST_CASE("load_plain_csv: in-memory / temporary files") {
    const std::filesystem::path tmp_plain = std::filesystem::temp_directory_path() / "test_plain.csv";
    {
        std::ofstream f(tmp_plain);
        f << "a,b\n1.0,2.0\n";
    }

    PlainCsv res = load_plain_csv(tmp_plain, "b");
    CHECK(res.X.rows() == 1);
    CHECK(res.X.cols() == 1);
    CHECK(res.X(0, 0) == doctest::Approx(1.0));
    CHECK(res.y.rows() == 1);
    CHECK(res.y.cols() == 1);
    CHECK(res.y(0, 0) == doctest::Approx(2.0));
    CHECK(res.column_names == std::vector<std::string>{"a"});

    // Header-only file
    const std::filesystem::path tmp_empty = std::filesystem::temp_directory_path() / "test_empty.csv";
    {
        std::ofstream f(tmp_empty);
        f << "a,b\n";
    }
    PlainCsv empty_res = load_plain_csv(tmp_empty, "b");
    CHECK(empty_res.X.rows() == 0);
    CHECK(empty_res.y.rows() == 0);
    CHECK(empty_res.column_names == std::vector<std::string>{"a"});

    // Ragged row throws with row number
    const std::filesystem::path tmp_ragged = std::filesystem::temp_directory_path() / "test_ragged.csv";
    {
        std::ofstream f(tmp_ragged);
        f << "a,b\n1.0\n";
    }
    try {
        load_plain_csv(tmp_ragged, "b");
        CHECK(false);
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        CHECK(msg.find("row 1") != std::string::npos);
    }

    std::filesystem::remove(tmp_plain);
    std::filesystem::remove(tmp_empty);
    std::filesystem::remove(tmp_ragged);
}

TEST_CASE("load_labeled_image_csv: simple temporary file") {
    const std::filesystem::path tmp_img = std::filesystem::temp_directory_path() / "test_img.csv";
    {
        std::ofstream f(tmp_img);
        f << "label,image\n0,\"[0.0, 1.0, 2.0]\"\n";
    }

    LabeledImageCsv res = load_labeled_image_csv(tmp_img);
    CHECK(res.labels.rows() == 1);
    CHECK(res.labels.cols() == 1);
    CHECK(res.labels(0, 0) == 0.0);
    CHECK(res.images.rows() == 1);
    CHECK(res.images.cols() == 3);
    CHECK(res.images(0, 0) == 0.0);
    CHECK(res.images(0, 1) == 1.0);
    CHECK(res.images(0, 2) == 2.0);

    // Mismatched image length in row 2
    {
        std::ofstream f(tmp_img);
        f << "label,image\n0,\"[0.0, 1.0, 2.0]\"\n1,\"[0.0, 1.0]\"\n";
    }
    try {
        load_labeled_image_csv(tmp_img);
        CHECK(false);
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        CHECK(msg.find("row 2") != std::string::npos);
    }

    std::filesystem::remove(tmp_img);
}

TEST_CASE("load_plain_csv: real fraud_dataset.csv verification") {
    const std::filesystem::path data_dir =
        std::filesystem::path(TP3_SOURCE_DIR) / "data" / "data and documentation";
    const std::filesystem::path fraud_path = data_dir / "fraud_dataset.csv";

    PlainCsv fraud = load_plain_csv(fraud_path, "flagged_fraud");
    CHECK(fraud.X.rows() == 7500);
    CHECK(fraud.y.rows() == 7500);
    CHECK(fraud.X.cols() == 10);
    CHECK(fraud.y.cols() == 1);

    // Row 0 assertions
    CHECK(fraud.X(0, 0) == doctest::Approx(1726079771.0));
    CHECK(fraud.X(0, 1) == doctest::Approx(42.07));
    CHECK(fraud.X(0, 2) == doctest::Approx(5.0));
    CHECK(fraud.X(0, 3) == doctest::Approx(422.6));
    CHECK(fraud.X(0, 4) == doctest::Approx(1.61));
    CHECK(fraud.X(0, 5) == doctest::Approx(830.0));
    CHECK(fraud.X(0, 6) == doctest::Approx(8297586.0));
    CHECK(fraud.X(0, 7) == doctest::Approx(1591.0));
    CHECK(fraud.X(0, 8) == doctest::Approx(9.0));
    CHECK(fraud.X(0, 9) == doctest::Approx(0.399304));
    CHECK(fraud.y(0, 0) == doctest::Approx(0.0));

    // Row 1 assertions
    CHECK(fraud.X(1, 0) == doctest::Approx(1712136533.0));
    CHECK(fraud.X(1, 1) == doctest::Approx(35.05));
    CHECK(fraud.X(1, 2) == doctest::Approx(6.0));
    CHECK(fraud.X(1, 3) == doctest::Approx(384.7));
    CHECK(fraud.X(1, 4) == doctest::Approx(23.44));
    CHECK(fraud.X(1, 5) == doctest::Approx(2993.0));
    CHECK(fraud.X(1, 6) == doctest::Approx(3682826.0));
    CHECK(fraud.X(1, 7) == doctest::Approx(4097.6));
    CHECK(fraud.X(1, 8) == doctest::Approx(10.0));
    CHECK(fraud.X(1, 9) == doctest::Approx(0.026173));
    CHECK(fraud.y(1, 0) == doctest::Approx(0.0));
}

TEST_CASE("load_labeled_image_csv: real digits datasets verification") {
    const std::filesystem::path data_dir =
        std::filesystem::path(TP3_SOURCE_DIR) / "data" / "data and documentation";

    SUBCASE("digits.csv") {
        const std::filesystem::path digits_path = data_dir / "digits.csv";
        LabeledImageCsv digits = load_labeled_image_csv(digits_path);
        CHECK(digits.labels.rows() == 12449);
        CHECK(digits.images.rows() == 12449);
        CHECK(digits.images.cols() == 784);

        // Row 0
        CHECK(digits.labels(0, 0) == 0.0);
        CHECK(digits.images(0, 0) == 0.0);
        double sum0 = 0.0;
        double max0 = 0.0;
        for (std::size_t j = 0; j < 784; ++j) {
            double v = digits.images(0, j);
            sum0 += v;
            if (v > max0) max0 = v;
        }
        CHECK(sum0 == doctest::Approx(123.5529420804).epsilon(1e-6));
        CHECK(max0 == doctest::Approx(1.0).epsilon(1e-6));

        // Row 1
        CHECK(digits.labels(1, 0) == 0.0);
        double sum1 = 0.0;
        for (std::size_t j = 0; j < 784; ++j) {
            sum1 += digits.images(1, j);
        }
        CHECK(sum1 == doctest::Approx(129.6392165795).epsilon(1e-6));
    }

    SUBCASE("digits_test.csv") {
        const std::filesystem::path test_path = data_dir / "digits_test.csv";
        LabeledImageCsv test_digits = load_labeled_image_csv(test_path);
        CHECK(test_digits.labels.rows() == 2497);
        CHECK(test_digits.images.rows() == 2497);
        CHECK(test_digits.images.cols() == 784);

        CHECK(test_digits.labels(0, 0) == 7.0);
        double sum0 = 0.0;
        for (std::size_t j = 0; j < 784; ++j) {
            sum0 += test_digits.images(0, j);
        }
        CHECK(sum0 == doctest::Approx(91.0784323998).epsilon(1e-6));
    }

    SUBCASE("more_digits.csv") {
        const std::filesystem::path more_path = data_dir / "more_digits.csv";
        LabeledImageCsv more_digits = load_labeled_image_csv(more_path);
        CHECK(more_digits.labels.rows() == 15741);
        CHECK(more_digits.images.rows() == 15741);
        CHECK(more_digits.images.cols() == 784);

        CHECK(more_digits.labels(0, 0) == 3.0);
        double sum0 = 0.0;
        for (std::size_t j = 0; j < 784; ++j) {
            sum0 += more_digits.images(0, j);
        }
        CHECK(sum0 == doctest::Approx(102.6078440305).epsilon(1e-6));
    }
}
