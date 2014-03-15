template <int N> class fnv;

    template <>
    class fnv<32>
    {
        static const uint32_t FNV_32_PRIME = 0x01000193UL;
        uint32_t m_offset;

    public:
        static const uint32_t INIT  = 0x811c9dc5UL;

        __device__ fnv(const uint32_t init = INIT)
        : m_offset(init) 
        {}

        __device__ void offset(uint32_t init)
        { m_offset = init; }

        __device__ uint32_t operator()(const std::string &_buf)
        {
            return operator()(_buf.c_str(), _buf.length());
        }

        __device__ uint32_t operator()(const char *_buf, size_t _len)
        {
            const unsigned char *bp = reinterpret_cast<const unsigned char *>(_buf); /* start of buffer */
            const unsigned char *be = bp + _len;                                     /* beyond end of buffer */

            uint32_t hval = m_offset;

            /*
             * FNV-1a hash each octet in the buffer
             */
            while (bp < be) {

                /* xor the bottom with the current octet */
                hval ^= (uint32_t)*bp++;

                /* multiply by the 32 bit FNV magic prime mod 2^32 */

#if defined(NO_FNV_GCC_OPTIMIZATION)
                hval *= FNV_32_PRIME;
#else
                hval += (hval<<1) + (hval<<4) + (hval<<7) + (hval<<8) + (hval<<24);
#endif
                if (hval == 0)
                    break;
            }

            /* return our new hash value */
            return m_offset = hval;
        }

    };

    template <>
    class fnv<64>
    {
        static const uint64_t FNV_64_PRIME = 0x100000001b3ULL;
        uint64_t m_offset;

    public:
        static const uint64_t INIT  = 0xcbf29ce484222325ULL;

        __device__ fnv(const uint64_t init = INIT)
        : m_offset(init) 
        {}

        __device__ void offset(uint64_t init)
        { m_offset = init; }

        __device__ uint64_t operator()(const std::string &_buf)
        {
            return operator()(_buf.c_str(), _buf.length());
        }

        __device__ uint64_t operator()(const char *_buf, size_t _len)
        {
            const unsigned char *bp = reinterpret_cast<const unsigned char *>(_buf); /* start of buffer */
            const unsigned char *be = bp + _len;                                     /* beyond end of buffer */

            uint64_t hval = m_offset;

            /*
             * FNV-1a hash each octet of the buffer
             */
            while (bp < be) {

                /* xor the bottom with the current octet */
                hval ^= (uint64_t)*bp++;

                /* multiply by the 64 bit FNV magic prime mod 2^64 */

#if defined(NO_FNV_GCC_OPTIMIZATION)
                hval *= FNV_64_PRIME;
#else
                hval += (hval << 1) + (hval << 4) + (hval << 5) +
                (hval << 7) + (hval << 8) + (hval << 40);
#endif
                if (hval == 0)
                    break;
            }

            return m_offset = hval;
        }
    };