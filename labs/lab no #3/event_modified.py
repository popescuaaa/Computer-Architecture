from threading import enumerate, Event, Thread, Condition

class Master(Thread):
    def __init__(self, max_work, work_condition):
        Thread.__init__(self, name = "Master")
        self.max_work = max_work
        # any condition needs a flag to perform
        self.work_condition = work_condition 
        self.flag = False

    def set_worker(self, worker):
        self.worker = worker
    
    def run(self):
        for i in range(self.max_work):
            # generate work
            self.work = i
            # notify worker
            self.flag = True
            self.work_condition.notify()

            # get result
            self.work_condition.wait()

            if self.get_work() + 1 != self.worker.get_result():
                print ("oops")
            print ("%d -> %d" % (self.work, self.worker.get_result()))
    
    def get_work(self):
        return self.work

class Worker(Thread):
    def __init__(self, terminate, work_condition):
        Thread.__init__(self, name = "Worker")
        self.terminate = terminate
        self.work_condition = work_condition

    def set_master(self, master):
        self.master = master
    
    def run(self):
        while(True):
            with self.work_condition:
                # wait work
                if not self.master.flag:
                    self.work_condition.wait()
                    self.master.work_condition = False

                if(terminate.is_set()): break
                # generate result
                self.result = self.master.get_work() + 1
                # notify master
                self.work_condition.notify()
    
    def get_result(self):
        return self.result

if __name__ ==  "__main__":
    # create shared objects
    terminate = Event()
    work_condition =  Condition()

    
    # start worker and master
    w = Worker(terminate, work_condition)
    m = Master(10, work_condition)
    w.set_master(m)
    m.set_worker(w)
    w.start()
    m.start()

    # wait for master
    m.join()

    # wait for worker
    with work_condition:
        terminate.set()
        work_condition.notifyAll()
    w.join()

    # print running threads for verification
    print(enumerate())
