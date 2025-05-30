//
// Generated file, do not edit! Created by nedtool 5.6 from veins/modules/application/traci/Request.msg.
//

// Disable warnings about unused variables, empty switch stmts, etc:
#ifdef _MSC_VER
#  pragma warning(disable:4101)
#  pragma warning(disable:4065)
#endif

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wshadow"
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wunused-parameter"
#  pragma clang diagnostic ignored "-Wc++98-compat"
#  pragma clang diagnostic ignored "-Wunreachable-code-break"
#  pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wshadow"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wold-style-cast"
#  pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#  pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif

#include <iostream>
#include <sstream>
#include <memory>
#include "Request_m.h"

namespace omnetpp {

// Template pack/unpack rules. They are declared *after* a1l type-specific pack functions for multiple reasons.
// They are in the omnetpp namespace, to allow them to be found by argument-dependent lookup via the cCommBuffer argument

// Packing/unpacking an std::vector
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::vector<T,A>& v)
{
    int n = v.size();
    doParsimPacking(buffer, n);
    for (int i = 0; i < n; i++)
        doParsimPacking(buffer, v[i]);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::vector<T,A>& v)
{
    int n;
    doParsimUnpacking(buffer, n);
    v.resize(n);
    for (int i = 0; i < n; i++)
        doParsimUnpacking(buffer, v[i]);
}

// Packing/unpacking an std::list
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::list<T,A>& l)
{
    doParsimPacking(buffer, (int)l.size());
    for (typename std::list<T,A>::const_iterator it = l.begin(); it != l.end(); ++it)
        doParsimPacking(buffer, (T&)*it);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::list<T,A>& l)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i = 0; i < n; i++) {
        l.push_back(T());
        doParsimUnpacking(buffer, l.back());
    }
}

// Packing/unpacking an std::set
template<typename T, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::set<T,Tr,A>& s)
{
    doParsimPacking(buffer, (int)s.size());
    for (typename std::set<T,Tr,A>::const_iterator it = s.begin(); it != s.end(); ++it)
        doParsimPacking(buffer, *it);
}

template<typename T, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::set<T,Tr,A>& s)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i = 0; i < n; i++) {
        T x;
        doParsimUnpacking(buffer, x);
        s.insert(x);
    }
}

// Packing/unpacking an std::map
template<typename K, typename V, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::map<K,V,Tr,A>& m)
{
    doParsimPacking(buffer, (int)m.size());
    for (typename std::map<K,V,Tr,A>::const_iterator it = m.begin(); it != m.end(); ++it) {
        doParsimPacking(buffer, it->first);
        doParsimPacking(buffer, it->second);
    }
}

template<typename K, typename V, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::map<K,V,Tr,A>& m)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i = 0; i < n; i++) {
        K k; V v;
        doParsimUnpacking(buffer, k);
        doParsimUnpacking(buffer, v);
        m[k] = v;
    }
}

// Default pack/unpack function for arrays
template<typename T>
void doParsimArrayPacking(omnetpp::cCommBuffer *b, const T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimPacking(b, t[i]);
}

template<typename T>
void doParsimArrayUnpacking(omnetpp::cCommBuffer *b, T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimUnpacking(b, t[i]);
}

// Default rule to prevent compiler from choosing base class' doParsimPacking() function
template<typename T>
void doParsimPacking(omnetpp::cCommBuffer *, const T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimPacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

template<typename T>
void doParsimUnpacking(omnetpp::cCommBuffer *, T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimUnpacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

}  // namespace omnetpp

namespace {
template <class T> inline
typename std::enable_if<std::is_polymorphic<T>::value && std::is_base_of<omnetpp::cObject,T>::value, void *>::type
toVoidPtr(T* t)
{
    return (void *)(static_cast<const omnetpp::cObject *>(t));
}

template <class T> inline
typename std::enable_if<std::is_polymorphic<T>::value && !std::is_base_of<omnetpp::cObject,T>::value, void *>::type
toVoidPtr(T* t)
{
    return (void *)dynamic_cast<const void *>(t);
}

template <class T> inline
typename std::enable_if<!std::is_polymorphic<T>::value, void *>::type
toVoidPtr(T* t)
{
    return (void *)static_cast<const void *>(t);
}

}

namespace veins {

// forward
template<typename T, typename A>
std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec);

// Template rule to generate operator<< for shared_ptr<T>
template<typename T>
inline std::ostream& operator<<(std::ostream& out,const std::shared_ptr<T>& t) { return out << t.get(); }

// Template rule which fires if a struct or class doesn't have operator<<
template<typename T>
inline std::ostream& operator<<(std::ostream& out,const T&) {return out;}

// operator<< for std::vector<T>
template<typename T, typename A>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec)
{
    out.put('{');
    for(typename std::vector<T,A>::const_iterator it = vec.begin(); it != vec.end(); ++it)
    {
        if (it != vec.begin()) {
            out.put(','); out.put(' ');
        }
        out << *it;
    }
    out.put('}');

    char buf[32];
    sprintf(buf, " (size=%u)", (unsigned int)vec.size());
    out.write(buf, strlen(buf));
    return out;
}

Register_Class(Request)

Request::Request(const char *name, short kind) : ::veins::BaseFrame1609_4(name, kind)
{
}

Request::Request(const Request& other) : ::veins::BaseFrame1609_4(other)
{
    copy(other);
}

Request::~Request()
{
}

Request& Request::operator=(const Request& other)
{
    if (this == &other) return *this;
    ::veins::BaseFrame1609_4::operator=(other);
    copy(other);
    return *this;
}

void Request::copy(const Request& other)
{
    this->senderAddress = other.senderAddress;
    this->serial = other.serial;
    this->idMovieWants = other.idMovieWants;
    this->receiverType = other.receiverType;
    this->idMovieNew = other.idMovieNew;
    this->durationNew = other.durationNew;
    this->demoData = other.demoData;
    this->idSender = other.idSender;
}

void Request::parsimPack(omnetpp::cCommBuffer *b) const
{
    ::veins::BaseFrame1609_4::parsimPack(b);
    doParsimPacking(b,this->senderAddress);
    doParsimPacking(b,this->serial);
    doParsimPacking(b,this->idMovieWants);
    doParsimPacking(b,this->receiverType);
    doParsimPacking(b,this->idMovieNew);
    doParsimPacking(b,this->durationNew);
    doParsimPacking(b,this->demoData);
    doParsimPacking(b,this->idSender);
}

void Request::parsimUnpack(omnetpp::cCommBuffer *b)
{
    ::veins::BaseFrame1609_4::parsimUnpack(b);
    doParsimUnpacking(b,this->senderAddress);
    doParsimUnpacking(b,this->serial);
    doParsimUnpacking(b,this->idMovieWants);
    doParsimUnpacking(b,this->receiverType);
    doParsimUnpacking(b,this->idMovieNew);
    doParsimUnpacking(b,this->durationNew);
    doParsimUnpacking(b,this->demoData);
    doParsimUnpacking(b,this->idSender);
}

const LAddress::L2Type& Request::getSenderAddress() const
{
    return this->senderAddress;
}

void Request::setSenderAddress(const LAddress::L2Type& senderAddress)
{
    this->senderAddress = senderAddress;
}

int Request::getSerial() const
{
    return this->serial;
}

void Request::setSerial(int serial)
{
    this->serial = serial;
}

int Request::getIdMovieWants() const
{
    return this->idMovieWants;
}

void Request::setIdMovieWants(int idMovieWants)
{
    this->idMovieWants = idMovieWants;
}

const char * Request::getReceiverType() const
{
    return this->receiverType.c_str();
}

void Request::setReceiverType(const char * receiverType)
{
    this->receiverType = receiverType;
}

int Request::getIdMovieNew() const
{
    return this->idMovieNew;
}

void Request::setIdMovieNew(int idMovieNew)
{
    this->idMovieNew = idMovieNew;
}

float Request::getDurationNew() const
{
    return this->durationNew;
}

void Request::setDurationNew(float durationNew)
{
    this->durationNew = durationNew;
}

const char * Request::getDemoData() const
{
    return this->demoData.c_str();
}

void Request::setDemoData(const char * demoData)
{
    this->demoData = demoData;
}

int Request::getIdSender() const
{
    return this->idSender;
}

void Request::setIdSender(int idSender)
{
    this->idSender = idSender;
}

class RequestDescriptor : public omnetpp::cClassDescriptor
{
  private:
    mutable const char **propertynames;
    enum FieldConstants {
        FIELD_senderAddress,
        FIELD_serial,
        FIELD_idMovieWants,
        FIELD_receiverType,
        FIELD_idMovieNew,
        FIELD_durationNew,
        FIELD_demoData,
        FIELD_idSender,
    };
  public:
    RequestDescriptor();
    virtual ~RequestDescriptor();

    virtual bool doesSupport(omnetpp::cObject *obj) const override;
    virtual const char **getPropertyNames() const override;
    virtual const char *getProperty(const char *propertyname) const override;
    virtual int getFieldCount() const override;
    virtual const char *getFieldName(int field) const override;
    virtual int findField(const char *fieldName) const override;
    virtual unsigned int getFieldTypeFlags(int field) const override;
    virtual const char *getFieldTypeString(int field) const override;
    virtual const char **getFieldPropertyNames(int field) const override;
    virtual const char *getFieldProperty(int field, const char *propertyname) const override;
    virtual int getFieldArraySize(void *object, int field) const override;

    virtual const char *getFieldDynamicTypeString(void *object, int field, int i) const override;
    virtual std::string getFieldValueAsString(void *object, int field, int i) const override;
    virtual bool setFieldValueAsString(void *object, int field, int i, const char *value) const override;

    virtual const char *getFieldStructName(int field) const override;
    virtual void *getFieldStructValuePointer(void *object, int field, int i) const override;
};

Register_ClassDescriptor(RequestDescriptor)

RequestDescriptor::RequestDescriptor() : omnetpp::cClassDescriptor(omnetpp::opp_typename(typeid(veins::Request)), "veins::BaseFrame1609_4")
{
    propertynames = nullptr;
}

RequestDescriptor::~RequestDescriptor()
{
    delete[] propertynames;
}

bool RequestDescriptor::doesSupport(omnetpp::cObject *obj) const
{
    return dynamic_cast<Request *>(obj)!=nullptr;
}

const char **RequestDescriptor::getPropertyNames() const
{
    if (!propertynames) {
        static const char *names[] = {  nullptr };
        omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
        const char **basenames = basedesc ? basedesc->getPropertyNames() : nullptr;
        propertynames = mergeLists(basenames, names);
    }
    return propertynames;
}

const char *RequestDescriptor::getProperty(const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? basedesc->getProperty(propertyname) : nullptr;
}

int RequestDescriptor::getFieldCount() const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? 8+basedesc->getFieldCount() : 8;
}

unsigned int RequestDescriptor::getFieldTypeFlags(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeFlags(field);
        field -= basedesc->getFieldCount();
    }
    static unsigned int fieldTypeFlags[] = {
        0,    // FIELD_senderAddress
        FD_ISEDITABLE,    // FIELD_serial
        FD_ISEDITABLE,    // FIELD_idMovieWants
        FD_ISEDITABLE,    // FIELD_receiverType
        FD_ISEDITABLE,    // FIELD_idMovieNew
        FD_ISEDITABLE,    // FIELD_durationNew
        FD_ISEDITABLE,    // FIELD_demoData
        FD_ISEDITABLE,    // FIELD_idSender
    };
    return (field >= 0 && field < 8) ? fieldTypeFlags[field] : 0;
}

const char *RequestDescriptor::getFieldName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldName(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldNames[] = {
        "senderAddress",
        "serial",
        "idMovieWants",
        "receiverType",
        "idMovieNew",
        "durationNew",
        "demoData",
        "idSender",
    };
    return (field >= 0 && field < 8) ? fieldNames[field] : nullptr;
}

int RequestDescriptor::findField(const char *fieldName) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    int base = basedesc ? basedesc->getFieldCount() : 0;
    if (fieldName[0] == 's' && strcmp(fieldName, "senderAddress") == 0) return base+0;
    if (fieldName[0] == 's' && strcmp(fieldName, "serial") == 0) return base+1;
    if (fieldName[0] == 'i' && strcmp(fieldName, "idMovieWants") == 0) return base+2;
    if (fieldName[0] == 'r' && strcmp(fieldName, "receiverType") == 0) return base+3;
    if (fieldName[0] == 'i' && strcmp(fieldName, "idMovieNew") == 0) return base+4;
    if (fieldName[0] == 'd' && strcmp(fieldName, "durationNew") == 0) return base+5;
    if (fieldName[0] == 'd' && strcmp(fieldName, "demoData") == 0) return base+6;
    if (fieldName[0] == 'i' && strcmp(fieldName, "idSender") == 0) return base+7;
    return basedesc ? basedesc->findField(fieldName) : -1;
}

const char *RequestDescriptor::getFieldTypeString(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeString(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldTypeStrings[] = {
        "veins::LAddress::L2Type",    // FIELD_senderAddress
        "int",    // FIELD_serial
        "int",    // FIELD_idMovieWants
        "string",    // FIELD_receiverType
        "int",    // FIELD_idMovieNew
        "float",    // FIELD_durationNew
        "string",    // FIELD_demoData
        "int",    // FIELD_idSender
    };
    return (field >= 0 && field < 8) ? fieldTypeStrings[field] : nullptr;
}

const char **RequestDescriptor::getFieldPropertyNames(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldPropertyNames(field);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    }
}

const char *RequestDescriptor::getFieldProperty(int field, const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldProperty(field, propertyname);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    }
}

int RequestDescriptor::getFieldArraySize(void *object, int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldArraySize(object, field);
        field -= basedesc->getFieldCount();
    }
    Request *pp = (Request *)object; (void)pp;
    switch (field) {
        default: return 0;
    }
}

const char *RequestDescriptor::getFieldDynamicTypeString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldDynamicTypeString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    Request *pp = (Request *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}

std::string RequestDescriptor::getFieldValueAsString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldValueAsString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    Request *pp = (Request *)object; (void)pp;
    switch (field) {
        case FIELD_senderAddress: {std::stringstream out; out << pp->getSenderAddress(); return out.str();}
        case FIELD_serial: return long2string(pp->getSerial());
        case FIELD_idMovieWants: return long2string(pp->getIdMovieWants());
        case FIELD_receiverType: return oppstring2string(pp->getReceiverType());
        case FIELD_idMovieNew: return long2string(pp->getIdMovieNew());
        case FIELD_durationNew: return double2string(pp->getDurationNew());
        case FIELD_demoData: return oppstring2string(pp->getDemoData());
        case FIELD_idSender: return long2string(pp->getIdSender());
        default: return "";
    }
}

bool RequestDescriptor::setFieldValueAsString(void *object, int field, int i, const char *value) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->setFieldValueAsString(object,field,i,value);
        field -= basedesc->getFieldCount();
    }
    Request *pp = (Request *)object; (void)pp;
    switch (field) {
        case FIELD_serial: pp->setSerial(string2long(value)); return true;
        case FIELD_idMovieWants: pp->setIdMovieWants(string2long(value)); return true;
        case FIELD_receiverType: pp->setReceiverType((value)); return true;
        case FIELD_idMovieNew: pp->setIdMovieNew(string2long(value)); return true;
        case FIELD_durationNew: pp->setDurationNew(string2double(value)); return true;
        case FIELD_demoData: pp->setDemoData((value)); return true;
        case FIELD_idSender: pp->setIdSender(string2long(value)); return true;
        default: return false;
    }
}

const char *RequestDescriptor::getFieldStructName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructName(field);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    };
}

void *RequestDescriptor::getFieldStructValuePointer(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructValuePointer(object, field, i);
        field -= basedesc->getFieldCount();
    }
    Request *pp = (Request *)object; (void)pp;
    switch (field) {
        case FIELD_senderAddress: return toVoidPtr(&pp->getSenderAddress()); break;
        default: return nullptr;
    }
}

} // namespace veins

