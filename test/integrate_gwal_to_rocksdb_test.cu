#include "rocksdb/db.h"
#include "rocksdb/transaction_log.h"
#include "gds/write_batch_generator.cuh"
#include "gds/gwal_writer.cuh"
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace rocksdb;

// Function to read file content as binary data
std::vector<char> readFileAsBinary(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filepath << std::endl;
        return {};
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    
    return buffer;
}

// Function to compare two files
bool compareFiles(const std::string& file1, const std::string& file2) {
    auto data1 = readFileAsBinary(file1);
    auto data2 = readFileAsBinary(file2);
    
    if (data1.empty() && data2.empty()) {
        return true; // Both files are empty or don't exist
    }
    
    if (data1.size() != data2.size()) {
        return false;
    }
    
    return std::equal(data1.begin(), data1.end(), data2.begin());
}

// Function to find files with specific pattern in directory
std::vector<std::string> findFiles(const std::string& dir, const std::string& pattern) {
    std::vector<std::string> result;
    
    if (!std::filesystem::exists(dir)) {
        std::cerr << "Directory does not exist: " << dir << std::endl;
        return result;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find(pattern) != std::string::npos) {
                result.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(result.begin(), result.end());
    return result;
}

int main() {

{
    Options options;
    options.create_if_missing = true;

    WriteOptions write_options;
    write_options.write_by_gds = true;

    std::string db_path = "./test_db_gwal";

    DB* db;
    Status status = DB::Open(options, db_path, &db);

    std::cout << "Open database successfully" << std::endl;


    SequenceNumber seq_num = db->GetLatestSequenceNumber();
    std::cout << "Latest sequence number: " << seq_num << std::endl;


    std::string current_wal_name = gpu_wal::GetWALPathName(db_path);



    gpu_wal::GWALWriter wal_writer(current_wal_name);
    gpu_wal::GPUWriteBatchGenerator generator(100, seq_num);
    
    if (!status.ok()) {
        std::cerr << "Cannot open database: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Open database successfully" << std::endl;
    
    std::string key = "test_key";
    std::string value = "hello_rocksdb";
    
    status = db->Put(write_options, key, value);

    gpu_wal::IOStatus s = generator.AddKeyValue(key, value);
    gpu_wal::WriteGeneratedBatchToWAL(generator, wal_writer);

    if (status.ok()) {
        std::cout << "PUT successfully: " << key << " -> " << value << std::endl;
    } else {
        std::cerr << "PUT failed: " << status.ToString() << std::endl;
    }
    
    // READ operation
    std::string read_value;
    status = db->Get(ReadOptions(), key, &read_value);
    if (status.ok()) {
        std::cout << "READ successfully: " << key << " -> " << read_value << std::endl;
    } else {
        std::cerr << "READ failed: " << status.ToString() << std::endl;
    }
    
    // Do some simple PUT/READ tests
    db->Put(write_options, "key1", "value1");
    s = generator.AddKeyValue("key1", "value1");
    gpu_wal::WriteGeneratedBatchToWAL(generator, wal_writer);
    db->Put(write_options, "key2", "value2");
    s = generator.AddKeyValue("key2", "value2");
    gpu_wal::WriteGeneratedBatchToWAL(generator, wal_writer);
    db->Put(write_options, "key3", "value3");
    s = generator.AddKeyValue("key3", "value3");
    gpu_wal::WriteGeneratedBatchToWAL(generator, wal_writer);
    
    std::string val;
    db->Get(ReadOptions(), "key1", &val);
    std::cout << "key1 -> " << val << std::endl;
    
    db->Get(ReadOptions(), "key2", &val);
    std::cout << "key2 -> " << val << std::endl;
    
    db->Get(ReadOptions(), "key3", &val);
    std::cout << "key3 -> " << val << std::endl;
    
    // Close database
    delete db;
    std::cout << "Close database successfully" << std::endl;
}


    // ================================ CPU mode ================================
{
    Options options;
    options.create_if_missing = true;

    WriteOptions write_options;
    write_options.write_by_gds = false;

    
    DB* db;
    Status status = DB::Open(options, "./test_db_cpu", &db);

    std::cout << "Open database successfully" << std::endl;
    
    if (!status.ok()) {
        std::cerr << "Cannot open database: " << status.ToString() << std::endl;
        return 1;
    }
    
    std::cout << "Open database successfully" << std::endl;
    
    std::string key = "test_key";
    std::string value = "hello_rocksdb";
    
    status = db->Put(write_options, key, value);


    if (status.ok()) {
        std::cout << "PUT successfully: " << key << " -> " << value << std::endl;
    } else {
        std::cerr << "PUT failed: " << status.ToString() << std::endl;
    }
    
    // READ operation
    std::string read_value;
    status = db->Get(ReadOptions(), key, &read_value);
    if (status.ok()) {
        std::cout << "READ successfully: " << key << " -> " << read_value << std::endl;
    } else {
        std::cerr << "READ failed: " << status.ToString() << std::endl;
    }
    
    // Do some simple PUT/READ tests
    db->Put(write_options, "key1", "value1");
    db->Put(write_options, "key2", "value2");
    db->Put(write_options, "key3", "value3");
    
    std::string val;
    db->Get(ReadOptions(), "key1", &val);
    std::cout << "key1 -> " << val << std::endl;
    
    db->Get(ReadOptions(), "key2", &val);
    std::cout << "key2 -> " << val << std::endl;
    
    db->Get(ReadOptions(), "key3", &val);
    std::cout << "key3 -> " << val << std::endl;
    
    // Close database
    delete db;
    std::cout << "Close database successfully" << std::endl;
}

    // // ================================ Compare DB Files ================================
    // std::cout << "\n=== Comparing GWAL DB and CPU DB files ===" << std::endl;
    
    // std::string gwal_db_path = "./test_db_gwal";
    // std::string cpu_db_path = "./test_db_cpu";
    
    // // Compare MANIFEST files
    // auto gwal_manifest_files = findFiles(gwal_db_path, "MANIFEST");
    // auto cpu_manifest_files = findFiles(cpu_db_path, "MANIFEST");
    
    // if (!gwal_manifest_files.empty() && !cpu_manifest_files.empty()) {
    //     std::string gwal_manifest = gwal_manifest_files.back();
    //     std::string cpu_manifest = cpu_manifest_files.back();
        
    //     bool manifest_same = compareFiles(gwal_manifest, cpu_manifest);
    //     std::cout << "MANIFEST files: " << (manifest_same ? "IDENTICAL" : "DIFFERENT") << std::endl;
    // } else {
    //     std::cout << "MANIFEST files: NOT FOUND" << std::endl;
    // }
    
    // // Compare LOG files
    // auto gwal_log_files = findFiles(gwal_db_path, ".log");
    // auto cpu_log_files = findFiles(cpu_db_path, ".log");
    
    // if (!gwal_log_files.empty() && !cpu_log_files.empty()) {
    //     std::string gwal_log = gwal_log_files.back();
    //     std::string cpu_log = cpu_log_files.back();
        
    //     bool log_same = compareFiles(gwal_log, cpu_log);
    //     std::cout << "LOG files: " << (log_same ? "IDENTICAL" : "DIFFERENT") << std::endl;
    // } else {
    //     std::cout << "LOG files: NOT FOUND" << std::endl;
    // }

    return 0;
}
