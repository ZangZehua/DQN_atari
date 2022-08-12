class MemoryBuffer(object):
    def __init__(self, memory_size=100000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def append(self, info):
        if len(self.buffer) <= self.memory_size:  # buffer not full
            self.buffer.append(info)
        else:  # buffer is full
            self.buffer[self.next_idx] = info
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def size(self):
        return len(self.buffer)

    