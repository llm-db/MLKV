#include <iostream>
#include <string>
#include <cassert>
#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"

using namespace rocksdb;

int main() {
    // Database instance
    DB* db;
    Options options;
    
    // Create the DB if it's not already present
    options.create_if_missing = true;
    
    // Open DB
    Status s = DB::Open(options, "./testdb", &db);
    assert(s.ok());
    
    std::cout << "RocksDB opened successfully!" << std::endl;
    
    // Put key-value pairs
    std::cout << "\n=== Putting key-value pairs ===" << std::endl;
    s = db->Put(WriteOptions(), "key1", "value1");
    assert(s.ok());
    std::cout << "Put: key1 -> value1" << std::endl;
    
    s = db->Put(WriteOptions(), "key2", "value2");
    assert(s.ok());
    std::cout << "Put: key2 -> value2" << std::endl;
    
    s = db->Put(WriteOptions(), "key3", "value3");
    assert(s.ok());
    std::cout << "Put: key3 -> value3" << std::endl;
    
    // Get values
    std::cout << "\n=== Getting values ===" << std::endl;
    std::string value;
    s = db->Get(ReadOptions(), "key1", &value);
    assert(s.ok());
    std::cout << "Get key1: " << value << std::endl;
    
    s = db->Get(ReadOptions(), "key2", &value);
    assert(s.ok());
    std::cout << "Get key2: " << value << std::endl;
    
    // Try to get a non-existent key
    s = db->Get(ReadOptions(), "nonexistent", &value);
    if (s.IsNotFound()) {
        std::cout << "Key 'nonexistent' not found (as expected)" << std::endl;
    }
    
    // Iterate through all key-value pairs
    std::cout << "\n=== Iterating through all keys ===" << std::endl;
    Iterator* it = db->NewIterator(ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::cout << "Key: " << it->key().ToString() 
                  << ", Value: " << it->value().ToString() << std::endl;
    }
    assert(it->status().ok());  // Check for any errors found during the scan
    delete it;
    
    // Delete a key
    std::cout << "\n=== Deleting key2 ===" << std::endl;
    s = db->Delete(WriteOptions(), "key2");
    assert(s.ok());
    std::cout << "Deleted key2" << std::endl;
    
    // Try to get the deleted key
    s = db->Get(ReadOptions(), "key2", &value);
    if (s.IsNotFound()) {
        std::cout << "Key 'key2' not found after deletion (as expected)" << std::endl;
    }
    
    // Show remaining keys
    std::cout << "\n=== Remaining keys after deletion ===" << std::endl;
    it = db->NewIterator(ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::cout << "Key: " << it->key().ToString() 
                  << ", Value: " << it->value().ToString() << std::endl;
    }
    assert(it->status().ok());
    delete it;
    
    // Batch operations
    std::cout << "\n=== Batch operations ===" << std::endl;
    WriteBatch batch;
    batch.Delete("key1");
    batch.Put("key4", "value4");
    batch.Put("key5", "value5");
    s = db->Write(WriteOptions(), &batch);
    assert(s.ok());
    std::cout << "Batch operation completed: deleted key1, added key4 and key5" << std::endl;
    
    // Show final state
    std::cout << "\n=== Final database state ===" << std::endl;
    it = db->NewIterator(ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::cout << "Key: " << it->key().ToString() 
                  << ", Value: " << it->value().ToString() << std::endl;
    }
    assert(it->status().ok());
    delete it;
    
    // Close database
    delete db;
    std::cout << "\nRocksDB closed successfully!" << std::endl;
    
    return 0;
}
